#!/usr/bin/env python3
"""Force L1/L2 eviction with distinct long 1h prefixes, then replay the oldest ones.

Reports decode pacing per phase and the server's tier counters, so an L3 change can be
judged in minutes: is decoding slower while L3 writes, do replays come back from storage,
does the scheduler stay healthy once idle.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kimi_probe import filler  # noqa: E402


def stream(base, model, system, max_tokens):
    body = {'model': model, 'max_tokens': max_tokens, 'stream': True, 'ignore_eos': True, 'temperature': 0.6,
            'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'enable_thinking': False},
            'prompt_cache_options': {'mode': 'implicit', 'ttl': '1h'},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Write a long story.'}]}
    request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    started, stamps, usage = time.time(), [], {}
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b'data:') or b'[DONE]' in line:
                continue
            event = json.loads(line[5:])
            if event.get('usage'):
                usage = event['usage']
            if event.get('choices') and event['choices'][0].get('delta', {}).get('content'):
                stamps.append(time.time())
    details = usage.get('prompt_tokens_details') or {}
    tokens = usage.get('completion_tokens', 0)
    return {'ttft': stamps[0] - started if stamps else None,
            'itl': (stamps[-1] - stamps[0]) / max(tokens - 1, 1) if len(stamps) > 1 else None,
            'read': details.get('cached_tokens', 0), 'write': details.get('cache_write_tokens', 0)}


def counters(base):
    with urllib.request.urlopen(base + '/metrics', timeout=30) as response:
        text = response.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith('qwen_ttl_tokens_total{'):
            out[line.split('kind="')[1].split('"')[0]] = float(line.rsplit(' ', 1)[1])
        elif line.startswith('sglang:prefill_effective_tokens_total{') and 'tp_rank="0"' in line:
            out['engine_' + line.split('mode="')[1].split('"')[0]] = float(line.rsplit(' ', 1)[1])
    return out


def phase(name, args, prefixes, repeat):
    before, started = counters(args.base), time.time()

    def one(text):
        return [stream(args.base, args.model, text, args.max_tokens) for _ in range(repeat)]

    with ThreadPoolExecutor(args.concurrency) as pool:
        results = [r for group in pool.map(one, prefixes) for r in group]
    after = counters(args.base)
    itl = [r['itl'] for r in results if r['itl']]
    ttft = [r['ttft'] for r in results if r['ttft']]
    summary = {'phase': name, 'requests': len(results), 'seconds': round(time.time() - started, 1),
               'itl_ms_mean': round(1000 * statistics.mean(itl), 1), 'itl_ms_p90': round(1000 * sorted(itl)[int(len(itl) * .9)], 1),
               'ttft_s_mean': round(statistics.mean(ttft), 2), 'billed_read': sum(r['read'] for r in results),
               'delta': {k: round(after.get(k, 0) - before.get(k, 0)) for k in after if after.get(k, 0) != before.get(k, 0)}}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='Qwen3.8-27B-FP8')
    parser.add_argument('--prefixes', type=int, default=60)
    parser.add_argument('--words', type=int, default=3000)
    parser.add_argument('--replay', type=int, default=10)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=120)
    parser.add_argument('--settle', type=float, default=180)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = f'{time.time():.0f}'
    prefixes = [f'stress {run} {i}. ' + filler(int(run) + i, args.words) for i in range(args.prefixes)]
    half = args.prefixes // 2
    out = [phase('fill-early', args, prefixes[:half // 2], 2), phase('fill-evicting', args, prefixes[half // 2:], 2)]
    time.sleep(args.settle)      # L3 writes are asynchronous; let them drain before reading back
    out.append(phase('replay-oldest', args, prefixes[:args.replay], 1))
    if args.output:
        args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
