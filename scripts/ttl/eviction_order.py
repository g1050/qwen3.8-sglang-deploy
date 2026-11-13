#!/usr/bin/env python3
"""Does capacity eviction take expired leases before valid ones, in every pool?

Short prefixes (--words 250) exhaust the state pool while KV stays mostly free; long ones
(--words 3000) fill KV first. Run both to cover both pools.

Needs a service started with QWEN_TTL_TEST_CLOCK=1. Group B (1h) is written first, group A
(5m) after it, then the clock moves past 5m: A is expired, B is valid but older in LRU
terms. A filler group C then forces eviction. Lease-aware eviction keeps B; plain LRU keeps A.
"""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acceptance import Service, prefix  # noqa: E402


ENGINE = {'sglang:kv_available_tokens': 'gpu_kv_free', 'sglang:mamba_available_tokens': 'gpu_state_free',
          'sglang:hicache_host_used_tokens': 'host_kv_used', 'sglang:hicache_host_total_tokens': 'host_kv_total'}


def snapshot(service):
    with urllib.request.urlopen(service.base + '/metrics', timeout=30) as response:
        text = response.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith('qwen_ttl_tokens_total{'):
            out[line.split('kind="')[1].split('"')[0]] = float(line.rsplit(' ', 1)[1])
        elif line.startswith('qwen_ttl_evictions_total{'):
            labels = dict(part.split('=') for part in line[line.index('{') + 1:line.index('}')].replace('"', '').split(','))
            out['evict {pool} {layer} {lease}'.format(**labels)] = float(line.rsplit(' ', 1)[1])
        elif line.startswith('sglang:') and 'tp_rank="0"' in line and line.split('{')[0] in ENGINE:
            out['engine ' + ENGINE[line.split('{')[0]]] = float(line.rsplit(' ', 1)[1])
        elif line.startswith('qwen_ttl_host_state_slots{'):
            out['host_state_' + line.split('state="')[1].split('"')[0]] = float(line.rsplit(' ', 1)[1])
    return out


def delta(after, before):
    return {k: round(v - before.get(k, 0)) for k, v in sorted(after.items())
            if v != before.get(k, 0) and not k.startswith(('host_state_', 'engine '))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='Qwen3.8-27B-FP8')
    parser.add_argument('--group', type=int, default=20)
    parser.add_argument('--filler', type=int, default=25)
    parser.add_argument('--words', type=int, default=3000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    service = Service(args.base, args.model)
    groups = {name: [prefix(args.words) for _ in range(count)]
              for name, count in (('B', args.group), ('A', args.group), ('C', args.filler))}
    report, mark = [], snapshot(service)

    def phase(name, texts, ttl, repeat):
        nonlocal mark
        read = 0
        for text in texts:
            for _ in range(repeat):
                service.now += 0.2
                read += service.call('chat', text, ttl=ttl)['bill']['read']
        now = snapshot(service)
        row = {'phase': name, 'billed_read': read, **delta(now, mark),
               'host_state_slots': [now.get('host_state_used'), now.get('host_state_total')],
               'pools': {k[7:]: round(v) for k, v in now.items() if k.startswith('engine ')}}
        mark = now
        report.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    phase('write B (1h)', groups['B'], '1h', 2)
    phase('write A (5m)', groups['A'], '5m', 2)
    service.now += 400
    phase('fill C (5m), A expired', groups['C'], '5m', 2)
    kept_b = phase('replay B (valid)', groups['B'], '1h', 1).get('physical', 0)
    kept_a = phase('replay A (expired)', groups['A'], '5m', 1).get('physical', 0)
    total = snapshot(service)
    evicted = {k: round(v) for k, v in total.items() if k.startswith('evict ')}
    verdict = {'physical_reuse_B_valid': kept_b, 'physical_reuse_A_expired': kept_a, 'evictions': evicted}
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    args.output.write_text(json.dumps({'phases': report, 'verdict': verdict}, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
