import random
import unittest

from qwen_ttl.core import (LeaseTable, PolicyError, ReferenceModel, bill, decode_carrier, encode_carrier,
                           page_keys, parse_options, readable_pages)

PAGE = 4


def tokens(seed, pages, tail=1):
    rng = random.Random(seed)
    return [rng.randrange(1000) for _ in range(pages * PAGE + tail)]


class System:
    """Lease table driven like the engine: physical data may outlive or predate its lease."""

    def __init__(self, keep=7200):
        self.table = LeaseTable(keep)
        self.present = set()

    def request(self, now, toks, ttl=300, salt=None, lose=False):
        keys = page_keys(toks, PAGE, salt)
        if lose:
            self.present -= set(keys)
        _, reusable = self.table.lengths(keys, now)
        used = 0
        for key in keys[:min(reusable, readable_pages(len(toks), PAGE))]:
            if key not in self.present:
                break
            used += 1
        read, tier, _ = self.table.admit(keys, used, ttl, now)
        self.present |= set(keys)
        return bill(len(toks), PAGE, read, tier), used


class Policy(unittest.TestCase):
    def test_defaults_and_validation(self):
        self.assertEqual(parse_options({}, 'chat'), 300)
        self.assertEqual(parse_options({'prompt_cache_options': {'mode': 'implicit'}}, 'chat'), 300)
        self.assertEqual(parse_options({'prompt_cache_options': {'ttl': '1h'}}, 'responses'), 3600)
        for bad in ({'ttl': '10m'}, {'ttl': 300}, {'mode': 'explicit', 'ttl': '5m'}, 'x'):
            with self.assertRaises(PolicyError):
                parse_options({'prompt_cache_options': bad}, 'chat')

    def test_messages_top_level_only(self):
        body = {'cache_control': {'type': 'ephemeral', 'ttl': '1h'},
                'system': [{'type': 'text', 'text': 'a', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}],
                'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'b', 'cache_control': {}}]}]}
        self.assertEqual(parse_options(body, 'messages'), 3600)
        self.assertNotIn('cache_control', json_text(body))
        self.assertEqual(parse_options({'system': [{'cache_control': {'ttl': '1h'}}]}, 'messages'), 300)
        with self.assertRaises(PolicyError):
            parse_options({'cache_control': {'type': 'ephemeral', 'ttl': '2h'}}, 'messages')

    def test_carrier_round_trip(self):
        self.assertEqual(decode_carrier(encode_carrier(3600, 12.5, 'org')), (3600, 12.5, 'org'))
        self.assertEqual(decode_carrier(encode_carrier(300, 1.0, None)), (300, 1.0, None))
        self.assertIsNone(decode_carrier('org'))
        with self.assertRaises(PolicyError):
            encode_carrier(300, 1.0, encode_carrier(300, 1.0, None))

    def test_keys_chain_and_isolation(self):
        a = tokens(1, 5)
        keys = page_keys(a, PAGE)
        self.assertEqual(len(keys), 5)
        changed = list(a)
        changed[2 * PAGE] += 1
        other = page_keys(changed, PAGE)
        self.assertEqual(keys[:2], other[:2])
        self.assertTrue(all(x != y for x, y in zip(keys[2:], other[2:])))
        self.assertNotEqual(keys[0], page_keys(a, PAGE, salt='org')[0])
        self.assertEqual(page_keys(a, PAGE, previous=keys[1], start=2), keys[2:])


def json_text(value):
    import json
    return json.dumps(value)


class Semantics(unittest.TestCase):
    def test_write_then_hit_then_expire(self):
        ref, a = ReferenceModel(PAGE), tokens(1, 10, tail=3)
        self.assertEqual(ref.request(0, a), {'read': 0, 'write_5m': 40, 'write_1h': 0, 'input': 3})
        self.assertEqual(ref.request(299, a), {'read': 40, 'write_5m': 0, 'write_1h': 0, 'input': 3})
        self.assertEqual(ref.request(598, a)['read'], 40)       # renewed at 299
        self.assertEqual(ref.request(898, a)['write_5m'], 40)   # now >= expires

    def test_tier_lock_and_inheritance(self):
        ref, a = ReferenceModel(PAGE), tokens(2, 10)
        longer = a[:-1] + tokens(3, 5)
        ref.request(0, a, ttl=300)
        got = ref.request(10, longer, ttl=3600)
        self.assertEqual((got['read'], got['write_5m'], got['write_1h']), (40, 20, 0))
        self.assertEqual(ref.request(320, longer, ttl=3600)['write_1h'], 60)   # fully expired: new tier
        self.assertEqual(ref.request(3000, longer, ttl=300)['read'], 60)       # locked 1h, renewed as 1h
        self.assertEqual(ref.request(6500, longer, ttl=300)['read'], 60)

    def test_aligned_prompt_keeps_last_page_out(self):
        ref, a = ReferenceModel(PAGE), tokens(4, 3, tail=0)
        ref.request(0, a)
        self.assertEqual(ref.request(1, a), {'read': 8, 'write_5m': 4, 'write_1h': 0, 'input': 0})


class Differential(unittest.TestCase):
    def test_expired_reuse_is_invisible(self):
        system, ref, a = System(), ReferenceModel(PAGE), tokens(5, 8)
        self.assertEqual(system.request(0, a)[0], ref.request(0, a))
        got, used = system.request(400, a, ttl=3600)
        self.assertEqual(used, 8)                      # physically reused
        self.assertEqual(got, ref.request(400, a, ttl=3600))
        self.assertEqual(got['read'], 0)
        self.assertEqual(system.request(3900, a)[0], ref.request(3900, a))

    def test_retention_bound(self):
        system, a = System(keep=600), tokens(6, 8)
        system.request(0, a)
        self.assertEqual(system.request(599, a, lose=False)[1], 8)
        self.assertEqual(system.request(599 + 600, a)[1], 0)   # idle >= keep: not reusable

    def test_output_pages_are_reusable_but_unbilled(self):
        table, keys = LeaseTable(), page_keys(tokens(7, 6), PAGE)
        table.admit(keys[:4], 0, 300, 0)
        table.retain(keys[4:], 0)
        self.assertEqual(table.lengths(keys, 1), (4, 6))
        self.assertEqual([table.rank(k, 1) for k in keys], [2, 2, 2, 2, 1, 1])
        self.assertEqual(table.rank(keys[0], 7200), 0)
        self.assertEqual(table.sweep(7300), 6)

    def test_random_sequences_match_reference(self):
        rng = random.Random(0)
        for case in range(200):
            system, ref, now = System(), ReferenceModel(PAGE), 0.0
            bases = [tokens(case * 10 + i, rng.randrange(1, 6)) for i in range(3)]
            for _ in range(40):
                now += rng.choice([1, 30, 299, 300, 301, 900, 3599, 3600, 3601, 8000])
                base = rng.choice(bases)
                toks = base[:-1] + tokens(rng.randrange(4), rng.randrange(0, 3)) if rng.random() < .5 else base
                ttl, salt = rng.choice([300, 3600]), rng.choice([None, None, 'org'])
                lose = rng.random() < .2 and not self.valid(ref, toks, salt, now)
                self.assertEqual(system.request(now, toks, ttl, salt, lose=lose)[0], ref.request(now, toks, ttl, salt))

    @staticmethod
    def valid(ref, toks, salt, now):
        return any(k in ref.leases and now < ref.leases[k][0] for k in page_keys(toks, PAGE, salt))


if __name__ == '__main__':
    unittest.main()
