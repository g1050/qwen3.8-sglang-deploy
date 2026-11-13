#!/usr/bin/env python3
"""Client disconnects under a full cache: does the scheduler's idle pool check stay clean?

Two benchmark runs ended with `pool memory leak detected` one second after the client
cancelled its in-flight requests. This fills the GPU pool, then repeatedly opens streaming
requests (old prefixes that need a reload, new ones mid-prefill, running decodes) and
drops the connections at random moments, with idle gaps so the idle check runs each round.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import random
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kimi_probe import filler  # noqa: E402


def body(model, system, max_tokens, ttl):
    data = {'model': model, 'max_tokens': max_tokens, 'stream': True, 'ignore_eos': True,
            'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Write a long story.'}]}
    if ttl:
        data['prompt_cache_options'] = {'mode': 'implicit', 'ttl': ttl}
    return json.dumps(data).encode()


def request(host, port, payload, abort_after=None):
    """Stream one request; drop the socket after `abort_after` seconds if given."""
    connection = http.client.HTTPConnection(host, port, timeout=900)
    try:
        connection.request('POST', '/v1/chat/completions', payload, {'Content-Type': 'application/json'})
        if abort_after is not None:
            connection.sock.settimeout(abort_after)
        started = time.time()
        response = connection.getresponse()
        while response.readline():
            if abort_after is not None and time.time() - started > abort_after:
                return 'aborted'
        return 'finished'
    except (TimeoutError, OSError):
        return 'aborted'
    finally:
        connection.close()


def healthy(host, port):
    try:
        with urllib.request.urlopen(f'http://{host}:{port}/health', timeout=20) as response:
            return response.status == 200
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model', default='Qwen3.8-27B-FP8')
    parser.add_argument('--prefixes', type=int, default=64)
    parser.add_argument('--words', type=int, default=3500)
    parser.add_argument('--rounds', type=int, default=40)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--ttl', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    run = int(time.time())
    texts = [f'storm {run} {i}. ' + filler(run + i, args.words) for i in range(args.prefixes)]
    with ThreadPoolExecutor(4) as pool:   # fill: every prefix twice so it is cached and backed up to L2
        list(pool.map(lambda text: [request(args.host, args.port, body(args.model, text, 24, args.ttl)) for _ in range(2)], texts))
    print(json.dumps({'phase': 'filled', 'healthy': healthy(args.host, args.port)}), flush=True)
    outcomes, failed_round = {'aborted': 0, 'finished': 0}, None
    for round_index in range(args.rounds):
        jobs = []
        for _ in range(args.concurrency):
            kind = rng.random()
            text = (rng.choice(texts[:args.prefixes // 2]) if kind < .5            # old: reload from L2/L3
                    else rng.choice(texts) + ' ' + filler(rng.randrange(10 ** 6), 1500) if kind < .8   # extends a cached prefix
                    else f'fresh {run} {round_index} {rng.random()} ' + filler(rng.randrange(10 ** 6), args.words))
            jobs.append((body(args.model, text, 400, args.ttl), rng.choice([0.05, 0.3, 1, 2, 4, 8, None])))
        with ThreadPoolExecutor(args.concurrency) as pool:
            for outcome in pool.map(lambda job: request(args.host, args.port, job[0], job[1]), jobs):
                outcomes[outcome] += 1
        time.sleep(rng.choice([1, 3, 6]))      # let the scheduler go idle and run its pool check
        if not healthy(args.host, args.port):
            failed_round = round_index
            break
    time.sleep(20)
    result = {'phase': 'storm', 'rounds_done': round_index + 1, 'failed_round': failed_round,
              'healthy': healthy(args.host, args.port), **outcomes}
    print(json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
