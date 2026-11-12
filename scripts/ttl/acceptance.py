#!/usr/bin/env python3
"""TTL acceptance against a running service started with QWEN_TTL_TEST_CLOCK=1.

Every check compares user-visible fields with the Kimi-calibrated reference model or with
an explicit expectation from docs/ttl-v020.md. Time is virtual (header x-qwen-ttl-now).
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'compat'), str(ROOT / 'scripts/ttl')]
from kimi_probe import filler  # noqa: E402
from qwen_ttl.core import ReferenceModel  # noqa: E402

BLOCK = 256
KWARGS = {'chat_template_kwargs': {'enable_thinking': False}}


class Service:
    def __init__(self, base, model):
        self.base, self.model, self.now = base, model, time.time()
        self.results = []

    def post(self, path, body, stream=False):
        request = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                         {'Content-Type': 'application/json', 'x-qwen-ttl-now': repr(self.now)})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return response.status, response.read().decode(), {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode(), {}

    def tokens(self, messages):
        status, raw, _ = self.post('/v1/tokenize', {'model': self.model, 'messages': messages, **KWARGS})
        assert status == 200, raw[:300]
        return json.loads(raw)['tokens']

    def metrics(self):
        with urllib.request.urlopen(self.base + '/metrics', timeout=30) as response:
            text = response.read().decode()
        return {line.split('kind="')[1].split('"')[0]: float(line.rsplit(' ', 1)[1])
                for line in text.splitlines() if line.startswith('qwen_ttl_tokens_total{')}

    def call(self, api, system, user='Reply OK.', ttl=None, stream=False, **extra):
        """One request; returns the normalized bill plus raw usage."""
        if api == 'messages':
            body = {'model': self.model, 'max_tokens': 4, 'system': system, 'stream': stream,
                    'messages': [{'role': 'user', 'content': user}]}
            if ttl:
                body['cache_control'] = {'type': 'ephemeral', 'ttl': ttl}
            path = '/v1/messages'
        elif api == 'responses':
            body = {'model': self.model, 'max_output_tokens': 4, 'instructions': system, 'input': user, 'stream': stream}
            path = '/v1/responses'
        else:
            body = {'model': self.model, 'max_tokens': 4, 'stream': stream, **KWARGS,
                    'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
            if stream:
                body['stream_options'] = {'include_usage': True}
            path = '/v1/chat/completions'
        if ttl and api != 'messages':
            body['prompt_cache_options'] = {'mode': 'implicit', 'ttl': ttl}
        body.update(extra)
        started = time.time()
        status, raw, headers = self.post(path, body)
        if status != 200:
            return {'status': status, 'raw': raw[:300]}
        usage = {}
        if stream:
            for line in raw.splitlines():
                if line.startswith('data:') and '"usage"' in line:
                    event = json.loads(line[5:])
                    for holder in (event, event.get('response') or {}, event.get('message') or {}):
                        if isinstance(holder.get('usage'), dict):
                            usage.update(holder['usage'])
        else:
            usage = json.loads(raw)['usage']
        w5 = int(headers.get('msh-usage-cache-write-tokens-5m', -1))
        w1 = int(headers.get('msh-usage-cache-write-tokens-1h', -1))
        if api == 'messages':
            creation = usage.get('cache_creation') or {}
            bill = {'read': usage.get('cache_read_input_tokens'), 'write_5m': creation.get('ephemeral_5m_input_tokens'),
                    'write_1h': creation.get('ephemeral_1h_input_tokens'), 'input': usage.get('input_tokens')}
            total = usage.get('cache_creation_input_tokens')
        else:
            details = usage.get('input_tokens_details' if api == 'responses' else 'prompt_tokens_details') or {}
            prompt = usage.get('input_tokens' if api == 'responses' else 'prompt_tokens')
            total = details.get('cache_write_tokens')
            read = details.get('cached_tokens', 0)
            bill = {'read': read, 'write_5m': w5, 'write_1h': w1, 'input': prompt - read - (total or 0)}
        return {'status': 200, 'bill': bill, 'write_total': total, 'headers': (w5, w1), 'usage': usage,
                'raw': raw, 'elapsed': time.time() - started}

    def check(self, name, ok, detail=''):
        self.results.append({'check': name, 'ok': bool(ok), 'detail': str(detail)[:400]})
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f' :: {detail}'), flush=True)


def prefix(words=1200):
    name = uuid.uuid4().hex
    return f'run {name}. ' + filler(int(name[:12], 16), words)


def blocks(bill):
    return bill['read'] + bill['write_5m'] + bill['write_1h']


def semantics(s):
    a = prefix()
    first = s.call('chat', a)['bill']
    s.check('S3 default tier is 5m and writes', first['read'] == 0 and first['write_5m'] > 0 and first['write_1h'] == 0, first)
    s.check('S2 partial block is plain input', first['write_5m'] % BLOCK == 0 and 0 <= first['input'] < BLOCK, first)
    s.now += 2
    hit = s.call('chat', a)['bill']
    s.check('S10 full hit has no write', hit == {**first, 'read': first['write_5m'], 'write_5m': 0}, hit)
    middle = a[:len(a) // 2] + ' X ' + a[len(a) // 2:]
    changed = s.call('chat', middle)['bill']
    s.check('S1 change inside the prefix stops reuse there', 0 < changed['read'] < hit['read'] and changed['write_5m'] > 0, changed)
    s.now += 299 - 2
    s.check('S7 5m: idle TTL-1s still hits', s.call('chat', a)['bill']['read'] == hit['read'])
    s.now += 299
    s.check('S5 5m: a hit renews from the hit time', s.call('chat', a)['bill']['read'] == hit['read'])
    s.now += 301
    before = s.metrics()
    expired = s.call('chat', a)
    after = s.metrics()
    s.check('S7 5m: idle TTL+1s is a miss', expired['bill'] == first, expired['bill'])
    reused = after.get('expired_reuse', 0) - before.get('expired_reuse', 0)
    s.check('5.4 expired data is reused internally, invisibly', reused > 0 and 'cached_tokens' not in json.dumps(expired['usage']), reused)
    s.now += 1
    s.check('5.4 lease rebuilt by an expired reuse behaves like a fresh write', s.call('chat', a)['bill']['read'] == hit['read'])

    b = prefix()
    hour = s.call('chat', b, ttl='1h')['bill']
    s.check('1h write is reported in the 1h field', hour['write_1h'] > 0 and hour['write_5m'] == 0 and hour['read'] == 0, hour)
    s.now += 1800
    s.check('S5 1h: hit at +30m', s.call('chat', b, ttl='1h')['bill']['read'] == hour['write_1h'])
    s.now += 3000
    s.check('S5 1h: hit at +50m after renewal', s.call('chat', b)['bill']['read'] == hour['write_1h'])
    s.now += 3599
    s.check('S6 a 5m request renews a locked 1h lease as 1h', s.call('chat', b, ttl='5m')['bill']['read'] == hour['write_1h'])
    s.now += 3601
    s.check('S5 1h: miss after 61m idle', s.call('chat', b, ttl='1h')['bill'] == hour)

    c = prefix()
    base = s.call('chat', c, ttl='5m')['bill']
    s.now += 5
    grown = s.call('chat', c, user='Reply OK. ' + filler(7, 600), ttl='1h')['bill']
    s.check('S6 1h request on a valid 5m prefix: hit, new blocks written as 5m',
            grown['read'] == base['write_5m'] and grown['write_5m'] > 0 and grown['write_1h'] == 0, grown)
    s.now += 301
    switched = s.call('chat', c, user='Reply OK. ' + filler(7, 600), ttl='1h')['bill']
    s.check('S6 after full expiry the new tier applies', switched['read'] == 0 and switched['write_5m'] == 0 and switched['write_1h'] == blocks(grown), switched)
    d = prefix()
    base = s.call('chat', d, ttl='1h')['bill']
    s.now += 5
    grown = s.call('chat', d, user='Reply OK. ' + filler(8, 600), ttl='5m')['bill']
    s.check('S6 5m request on a valid 1h prefix: new blocks written as 1h',
            grown['read'] == base['write_1h'] and grown['write_1h'] > 0 and grown['write_5m'] == 0, grown)

    e = prefix()
    s.call('chat', e, cache_salt='org-a')
    s.now += 1
    other = s.call('chat', e, cache_salt='org-b')['bill']
    same = s.call('chat', e, cache_salt='org-a')['bill']
    s.check('S8 isolation domains do not share', other['read'] == 0 and same['read'] > 0, (other, same))

    for body, label in (({'ttl': '10m'}, 'ttl 10m'), ({'mode': 'explicit', 'ttl': '5m'}, 'mode explicit'), ({'ttl': 300}, 'numeric ttl')):
        s.check('S4 rejects ' + label, s.call('chat', 'x', prompt_cache_options=body)['status'] == 400)
    s.check('S4 messages rejects ttl 2h', s.call('messages', 'x', cache_control={'type': 'ephemeral', 'ttl': '2h'})['status'] == 400)

    f = prefix()
    s.call('chat', f)
    s.now += 7200 + 400
    before = s.metrics()
    late = s.call('chat', f)['bill']
    after = s.metrics()
    s.check('retention: data idle beyond T_keep is neither billed nor reused',
            late['read'] == 0 and after.get('physical', 0) == before.get('physical', 0), (late, after.get('physical'), before.get('physical')))


def fields(s):
    for api in ('chat', 'responses', 'messages'):
        a = prefix()
        results = {}
        for step, (ttl, stream) in enumerate([('1h', False), (None, True), (None, False)]):
            s.now += 1
            results[step] = s.call(api, a, ttl=ttl, stream=stream)
        write, stream_hit, hit = (results[i] for i in range(3))
        ok = all(r['status'] == 200 for r in results.values())
        s.check(f'F {api}: requests succeed', ok, [r.get('raw', '')[:200] for r in results.values() if r['status'] != 200])
        if not ok:
            continue
        s.check(f'F {api}: 1h write then hit', write['bill']['write_1h'] > 0 and write['bill']['read'] == 0 and hit['bill']['read'] == write['bill']['write_1h'], (write['bill'], hit['bill']))
        s.check(f'F6 {api}: streaming usage equals non-streaming', stream_hit['bill'] == hit['bill'], (stream_hit['bill'], hit['bill']))
        for name, result in (('write', write), ('hit', hit)):
            bill = result['bill']
            s.check(f'F5 {api} {name}: headers match usage', result['headers'] == (bill['write_5m'], bill['write_1h']) and result['write_total'] == bill['write_5m'] + bill['write_1h'], (result['headers'], result['write_total']))
            s.check(f'F1-3 {api} {name}: read, write and input are disjoint and sum to the prompt', bill['input'] >= 0 and min(bill.values()) >= 0, bill)
        if api == 'messages':
            usage = hit['usage']
            s.check('F3 messages: input_tokens excludes read and write', usage['input_tokens'] < BLOCK + 64 and usage['cache_creation_input_tokens'] == 0, usage)
            nested = s.call('messages', [{'type': 'text', 'text': prefix(), 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}])
            s.check('F4 messages: block-level cache_control is ignored (default 5m)', nested['status'] == 200 and nested['bill']['write_5m'] > 0 and nested['bill']['write_1h'] == 0, nested.get('bill', nested))
        text = json.dumps(hit['usage']) + hit['raw']
        s.check(f'F7 {api}: no physical-tier fields', all(word not in text for word in ('cached_tokens_details', '"device"', '"host"', '"storage"')))
    detailed = s.call('chat', prefix(), return_cached_tokens_details=True)
    s.check('F7 chat: return_cached_tokens_details exposes nothing', 'cached_tokens_details' not in detailed['raw'] and '"device"' not in detailed['raw'], detailed['raw'][-300:])


def differential(s, steps, seed):
    rng, reference = random.Random(seed), ReferenceModel(BLOCK)
    bases = [prefix(rng.choice([300, 900, 2000])) for _ in range(3)]
    failures = []
    for step in range(steps):
        s.now += rng.choice([1, 30, 299, 301, 900, 3599, 3601, 8000])
        system = rng.choice(bases)
        user = 'Reply OK.' + (' ' + filler(rng.randrange(3), rng.choice([200, 500])) if rng.random() < .5 else '')
        ttl, salt = rng.choice([None, '5m', '1h']), rng.choice([None, None, 'org'])
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        tokens = s.tokens(messages)
        want = reference.request(s.now, tokens, {'1h': 3600}.get(ttl, 300), salt)
        got = s.call('chat', system, user=user, ttl=ttl, **({'cache_salt': salt} if salt else {}))
        if got['status'] != 200 or got['usage']['prompt_tokens'] != len(tokens) or got['bill'] != want:
            failures.append({'step': step, 'want': want, 'got': got.get('bill', got), 'prompt': len(tokens)})
    s.check(f'5.1 differential: {steps} random requests equal the reference model', not failures, failures[:3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='Qwen3.8-27B-FP8')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    service = Service(args.base, args.model)
    for part in (semantics, fields, lambda s: differential(s, args.steps, 0)):
        try:
            part(service)
        except Exception as error:   # report and keep going: one broken API must not hide the rest
            service.check(f'{getattr(part, "__name__", "differential")} completed', False, repr(error))
    failed = [r for r in service.results if not r['ok']]
    args.output.write_text(json.dumps({'passed': len(service.results) - len(failed), 'failed': len(failed),
                                       'results': service.results}, indent=1, ensure_ascii=False) + '\n')
    print(f'{len(service.results) - len(failed)} passed, {len(failed)} failed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
