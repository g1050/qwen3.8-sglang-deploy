#!/usr/bin/env python3
"""Replay timed cache-probe sequences against an endpoint and log every usage field and header.

Used to calibrate the TTL reference model against the official Kimi endpoint and to run the
same sequences against the local service. The key is read from a private file, never logged.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

WORDS = ('alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november '
         'oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu').split()


def filler(seed, words):
    """Deterministic, non-repeating text so distinct prefixes never share cache blocks."""
    state, out = seed, []
    for index in range(words):
        state = (state * 6364136223846793005 + 1442695040888963407) % 2 ** 64
        out.append(f'{WORDS[(state >> 33) % len(WORDS)]}{index}')
    return ' '.join(out)


def call(base, key, api, model, system, user, ttl, stream=False):
    headers = {'Content-Type': 'application/json'}
    if api == 'messages':
        url = base + MESSAGES_PATH[0]
        headers.update({'x-api-key': key, 'anthropic-version': '2023-06-01'})
        body = {'model': model, 'max_tokens': 8, 'system': system, 'messages': [{'role': 'user', 'content': user}]}
        if ttl:
            body['cache_control'] = {'type': 'ephemeral', 'ttl': ttl}
    else:
        headers['Authorization'] = 'Bearer ' + key
        if api == 'responses':
            url = base + '/v1/responses'
            body = {'model': model, 'max_output_tokens': 8, 'instructions': system, 'input': user}
        else:
            url = base + '/v1/chat/completions'
            body = {'model': model, 'max_tokens': 8, 'stream': stream,
                    'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
            if stream:
                body['stream_options'] = {'include_usage': True}
        if ttl:
            body['prompt_cache_options'] = {'mode': 'implicit', 'ttl': ttl}
    body.update(EXTRA)
    request = urllib.request.Request(url, json.dumps(body).encode(), headers)
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw, status, got = response.read().decode(), response.status, dict(response.headers)
    except urllib.error.HTTPError as error:
        raw, status, got = error.read().decode(), error.code, dict(error.headers)
    usage = None
    if stream and status == 200:
        for line in raw.splitlines():
            if line.startswith('data:') and '"usage"' in line:
                usage = json.loads(line[5:]).get('usage') or usage
    else:
        try:
            parsed = json.loads(raw)
            usage = parsed.get('usage') if status == 200 else parsed
        except ValueError:
            usage = raw[:500]
    return {'status': status, 'usage': usage, 'elapsed': round(time.time() - started, 3),
            'headers': {k: v for k, v in got.items() if 'usage' in k.lower() or 'cache' in k.lower()}}


EXTRA = {}
MESSAGES_PATH = ['/anthropic/v1/messages']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sequence', type=Path, required=True, help='JSON list of steps')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--messages-path', default='/anthropic/v1/messages')
    parser.add_argument('--retry-wait', type=float, default=0)
    parser.add_argument('--extra', default='{}', help='JSON merged into every request body')
    args = parser.parse_args()
    EXTRA.update(json.loads(args.extra))
    MESSAGES_PATH[0] = args.messages_path
    key = args.key_file.read_text().strip() if args.key_file else 'none'
    nonce = uuid.uuid4().hex
    started = time.time()
    with args.output.open('a') as log:
        for step in json.loads(args.sequence.read_text()):
            time.sleep(step.get('wait', 0))
            # prefix: named deterministic body behind a per-run nonce; ext appends more words.
            seed = sum(map(ord, step['prefix']))
            system = f'run {nonce} {step["prefix"]}. ' + filler(seed, step.get('words', 3000))
            if step.get('mutate'):
                system = system.replace(' ', ' X ', 1) if step['mutate'] == 'head' else system[:len(system) // 2] + ' X ' + system[len(system) // 2:]
            user = 'Reply OK. ' + filler(seed + 1, step.get('ext', 0)) if step.get('ext') else 'Reply OK.'
            for _ in range(20):
                result = call(args.base, key, step.get('api', 'chat'), args.model, system, user, step.get('ttl'), step.get('stream', False))
                if result['status'] != 429:
                    break
                time.sleep(args.retry_wait)
            record = {'t': round(time.time() - started, 1), 'step': step, **result}
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
            log.flush()
            print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
