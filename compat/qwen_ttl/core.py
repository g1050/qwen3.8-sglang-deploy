"""Request policy, lease table and reference model. No engine imports.

A lease belongs to a chained page hash, not to a cache tier. `last_used` bounds physical
reuse (retention), `expires` bounds billing hits. A missing row means "not reusable".
"""
from array import array
import base64
import hashlib
import json
import os
import sys

TIERS = {'5m': 300, '1h': 3600}
# Kimi's default is 5m; the override exists so a benchmark client that sends no option can exercise 1h.
DEFAULT_TTL = int(os.environ.get('QWEN_TTL_DEFAULT_SECONDS', '300'))
assert DEFAULT_TTL in (300, 3600)
CARRIER = 'qwen-ttl-v2:'


class PolicyError(ValueError):
    pass


def parse_options(body, api):
    """Pop the vendor cache option from a request body and return the TTL in seconds."""
    if api == 'messages':
        control = body.pop('cache_control', None)
        strip_nested_cache_control(body)
        if control is None:
            return DEFAULT_TTL
        if not isinstance(control, dict) or control.get('type') != 'ephemeral':
            raise PolicyError('cache_control.type must be ephemeral')
        ttl = control.get('ttl', '5m')
    else:
        options = body.pop('prompt_cache_options', None)
        if options is None:
            return DEFAULT_TTL
        if not isinstance(options, dict):
            raise PolicyError('prompt_cache_options must be an object')
        if options.get('mode', 'implicit') != 'implicit':
            raise PolicyError('prompt_cache_options.mode only supports implicit')
        ttl = options.get('ttl', '5m')
    if not isinstance(ttl, str) or ttl not in TIERS:
        raise PolicyError('cache ttl only supports 5m and 1h')
    return TIERS[ttl]


def strip_nested_cache_control(value):
    """Block-level cache_control markers are ignored; only the top-level field applies."""
    if isinstance(value, dict):
        for key in ('system', 'messages', 'tools'):
            strip_blocks(value.get(key))


def strip_blocks(value):
    if isinstance(value, list):
        for item in value:
            strip_blocks(item)
    elif isinstance(value, dict):
        value.pop('cache_control', None)
        strip_blocks(value.get('content'))


def encode_carrier(seconds, stamp, salt):
    """Transport TTL and the HTTP arrival time to every scheduler rank inside cache_salt."""
    if isinstance(salt, str) and salt.startswith(CARRIER):
        raise PolicyError('Reserved cache_salt namespace')
    raw = json.dumps([seconds, stamp, salt or None], separators=(',', ':'), ensure_ascii=False).encode()
    return CARRIER + base64.urlsafe_b64encode(raw).decode()


def decode_carrier(salt):
    if not isinstance(salt, str) or not salt.startswith(CARRIER):
        return None
    seconds, stamp, original = json.loads(base64.urlsafe_b64decode(salt[len(CARRIER):]))
    return int(seconds), float(stamp), original


def page_keys(tokens, page_size, salt=None, extra_key=None, previous=None, start=0):
    """Chained 128-bit hashes of complete pages; `previous`/`start` continue an earlier chain."""
    if previous is None:
        seed = json.dumps(['qwen-ttl-pages-v2', page_size, salt, extra_key], ensure_ascii=False, separators=(',', ':'))
        previous = hashlib.blake2b(seed.encode(), digest_size=16).digest()
    data = tokens if isinstance(tokens, array) and tokens.typecode == 'q' else array('q', tokens)
    if sys.byteorder != 'little':
        data = array('q', data)
        data.byteswap()
    view = memoryview(data).cast('B')
    width = page_size * 8
    keys = []
    for offset in range(start * width, len(view) - width + 1, width):
        previous = hashlib.blake2b(view[offset:offset + width], digest_size=16, key=previous).digest()
        keys.append(previous)
    return keys


class LeaseTable:
    """key -> (last_used, expires, ttl). Every rank holds an identical copy."""

    def __init__(self, keep_seconds=7200):
        self.keep = keep_seconds
        self.rows = {}
        self.swept = 0.0

    def lengths(self, keys, now):
        """(billing pages, reusable pages): leading runs of valid / retained leases."""
        billing = reusable = 0
        rows, keep = self.rows, self.keep
        counting = True
        for key in keys:
            row = rows.get(key)
            if row is None or now >= row[0] + keep:
                break
            reusable += 1
            if counting and now < row[1]:
                billing += 1
            else:
                counting = False
        return billing, reusable

    def admit(self, keys, used_pages, ttl, now):
        """A request starts using `used_pages` physical pages of its prompt.

        Returns (read pages, tier of everything written). Hits renew at their locked tier;
        the rest of the prompt gets a lease whose tier is inherited from a valid prefix.
        """
        billing, _ = self.lengths(keys, now)
        read = min(billing, used_pages)
        rows = self.rows
        tier = rows[keys[read - 1]][2] if read else ttl
        for key in keys[:read]:
            locked = rows[key][2]
            rows[key] = (now, now + locked, locked)
        for key in keys[read:]:
            rows[key] = (now, now + tier, tier)
        return read, tier, billing

    def retain(self, keys, now):
        """Pages that were computed but never billed (generated output): reusable, not a hit."""
        rows = self.rows
        for key in keys:
            row = rows.get(key)
            if row is None or now >= row[1]:
                rows[key] = (now, 0.0, row[2] if row else 0)

    def rank(self, key, now):
        """Eviction class: 0 beyond retention, 1 expired but retained, 2 valid."""
        row = self.rows.get(key)
        if row is None or now >= row[0] + self.keep:
            return 0
        return 2 if now < row[1] else 1

    def sweep(self, now, interval=60):
        if now - self.swept < interval:
            return 0
        self.swept = now
        dead = [key for key, row in self.rows.items() if now >= row[0] + self.keep]
        for key in dead:
            del self.rows[key]
        return len(dead)


def bill(prompt_tokens, page_size, read_pages, tier):
    """User-visible accounting: read + write(5m) + write(1h) + uncached tail == prompt."""
    read = read_pages * page_size
    write = prompt_tokens // page_size * page_size - read
    return {'read': read, 'write_5m': write if tier == 300 else 0, 'write_1h': write if tier == 3600 else 0,
            'input': prompt_tokens - read - write}


def readable_pages(prompt_tokens, page_size):
    """Every complete block can be billed as read; the engine recomputing the last token is internal."""
    return prompt_tokens // page_size


class ReferenceModel:
    """Public semantics only: data exists exactly while its lease is valid. The test oracle."""

    def __init__(self, page_size):
        self.page_size = page_size
        self.leases = {}

    def request(self, now, tokens, ttl=DEFAULT_TTL, salt=None, extra_key=None):
        keys = page_keys(tokens, self.page_size, salt, extra_key)
        read = 0
        for key in keys[:readable_pages(len(tokens), self.page_size)]:
            lease = self.leases.get(key)
            if lease is None or now >= lease[0]:
                break
            read += 1
        tier = self.leases[keys[read - 1]][1] if read else ttl
        for key in keys[:read]:
            locked = self.leases[key][1]
            self.leases[key] = (now + locked, locked)
        for key in keys[read:]:
            self.leases[key] = (now + tier, tier)
        return bill(len(tokens), self.page_size, read, tier)
