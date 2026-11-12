#!/usr/bin/env python3
"""Render, start and inspect the Qwen3.8-27B-FP8 SGLang service (Docker).

Profiles:
  service      real decoding, the default external service (TTL and L3 off by default)
  performance  InferenceX AgentX口径: native MTP + simulated acceptance, benchmark only
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'
CFG = json.loads((ROOT / 'config/deployment.json').read_text())
HOST = json.loads((ROOT / 'config/host.json').read_text())
sys.path.insert(0, str(ROOT / 'scripts'))
from cache_profile import apply_profile  # noqa: E402


def run(argv, **kwargs):
    print(shlex.join(map(str, argv)), flush=True)
    kwargs.setdefault('check', True)
    return subprocess.run(list(map(str, argv)), **kwargs)


def docker(*args, **kwargs):
    return run(['docker', *args], **kwargs)


def container_name(profile):
    return CFG['container_name'] if profile == 'service' else CFG['performance_container_name']


def materialize(profile='service'):
    args, env = dict(CFG['args']), dict(CFG['env'])
    acceptance = None
    if profile == 'performance':
        perf = CFG['performance']
        acceptance = perf['acceptance']
        args.update(perf['args'])
        if perf['args']['speculative-num-steps'] != acceptance['num_speculative_tokens']:
            raise ValueError('Simulated acceptance was selected for a different draft length')
        env.update(SGLANG_SIMULATE_ACC_LEN=str(acceptance['acceptance_length']),
                   SGLANG_SIMULATE_ACC_METHOD=acceptance['method'],
                   SGLANG_SIMULATE_ACC_TOKEN_MODE=acceptance['token_mode'])
    args, env, cache = apply_profile(CFG, args, env, json.loads((ROOT / 'config/cache.json').read_text()))
    argv = ['python3', '/compat/launch.py'] if cache['ttl_enabled'] else ['python3', '-m', 'sglang.launch_server']
    for key, value in args.items():
        if value is True:
            argv.append('--' + key)
        elif value is not False and value is not None:
            argv += ['--' + key, str(value)]
    return {'profile': profile, 'version': CFG['version'], 'image': CFG['image'], 'image_id': CFG['image_id'],
            'container_name': container_name(profile), 'port': CFG['port'],
            'acceptance': acceptance or 'real; no speculative decoding, no SGLANG_SIMULATE_ACC variables',
            'cache': cache, 'env': env, 'argv': argv}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while block := stream.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(path, full_hash=False):
    expected = json.loads((ROOT / 'reference/checkpoint-manifest.json').read_text())
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    if set(shards) != set(expected):
        raise ValueError('Checkpoint shard names do not match the recorded manifest')
    for name in shards:
        file = path / name
        if not file.is_file() or file.stat().st_size != expected[name]['bytes']:
            raise ValueError(f'Missing or wrong-size checkpoint shard: {name}')
        with file.open('rb') as stream:
            size = int.from_bytes(stream.read(8), 'little')
            header = json.loads(stream.read(size))
        end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        if file.stat().st_size != 8 + size + end:
            raise ValueError(f'Truncated shard: {name}')
        if full_hash:
            if sha256_file(file) != expected[name]['sha256']:
                raise ValueError(f'Checkpoint SHA256 mismatch: {name}')
            print(f'{name}: SHA256 verified', flush=True)
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        if not (path / name).is_file():
            raise ValueError(f'Missing {name}')
    return {'shards': len(shards), 'bytes': sum(expected[n]['bytes'] for n in shards),
            'config_sha256': hashlib.sha256((path / 'config.json').read_bytes()).hexdigest()}


def port_owner():
    out = subprocess.check_output(['docker', 'ps', '--filter', f"publish={CFG['port']}", '--format', '{{.Names}}'], text=True)
    return out.split()


def start(profile):
    record = materialize(profile)
    record['checkpoint'] = checkpoint(Path(HOST['model_dir']))
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', CFG['image']], text=True).strip()
    if image_id != CFG['image_id']:
        raise ValueError(f'Image id mismatch: {image_id}')
    if owners := port_owner():
        raise ValueError(f"Port {CFG['port']} is published by running container(s) {owners}; stop them first. "
                         'This project never stops a container it did not start.')
    cache = Path(HOST['cache_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    if record['cache'].get('container_storage_dir'):
        # Pre-create once before rank startup; native mkdir is not synchronized.
        (cache / record['cache']['container_storage_dir'].removeprefix('/cache/')).mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(exist_ok=True)
    (RUNTIME / 'effective-deployment.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    args = ['run', '-d', '--name', record['container_name'], '--init', '--ipc=host', '--gpus=all',
            '-p', f"{CFG['port']}:{CFG['port']}",
            '--ulimit=memlock=-1:-1', '--log-driver=json-file', '--log-opt=max-size=100m', '--log-opt=max-file=3',
            '-v', f"{HOST['model_dir']}:/model:ro", '-v', f'{cache}:/cache:rw']
    if profile == 'service':
        args.append('--restart=unless-stopped')
    if record['cache']['ttl_enabled']:
        args += ['-v', f"{ROOT / 'compat'}:/compat:ro"]
    for key, value in record['env'].items():
        args += ['-e', f'{key}={value}']
    docker(*args, '--entrypoint=python3', CFG['image_id'], *record['argv'][1:])


def wait(profile, timeout=3600):
    name = container_name(profile)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = subprocess.run(['docker', 'inspect', '--format', '{{.State.Status}}', name], capture_output=True, text=True)
        if state.stdout.strip() != 'running':
            raise RuntimeError(f'Container {name} is not running: {state.stdout.strip() or state.stderr.strip()}')
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{CFG['port']}/health", timeout=5) as response:
                if response.status == 200:
                    print('Server ready')
                    return
        except OSError:
            pass
        time.sleep(10)
    raise TimeoutError(f'Server did not become ready within {timeout} seconds')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', choices=['service', 'performance'], default='service')
    parser.add_argument('action', choices=['render', 'check', 'verify-weights', 'start', 'stop', 'remove', 'logs', 'status', 'wait'])
    args = parser.parse_args()
    name = container_name(args.profile)
    if args.action == 'render':
        print(json.dumps(materialize(args.profile), indent=2, ensure_ascii=False))
    elif args.action == 'check':
        print(json.dumps(checkpoint(Path(HOST['model_dir'])), indent=2))
    elif args.action == 'verify-weights':
        print(json.dumps(checkpoint(Path(HOST['model_dir']), full_hash=True), indent=2))
    elif args.action == 'start':
        start(args.profile)
    elif args.action == 'stop':
        docker('stop', '--time=60', name)
    elif args.action == 'remove':
        # Docker refuses to remove a running container without --force.
        docker('rm', name)
    elif args.action == 'logs':
        docker('logs', '--tail=100', name)
    elif args.action == 'status':
        docker('ps', '-a', '--filter', f'name={name}')
    elif args.action == 'wait':
        wait(args.profile)


if __name__ == '__main__':
    main()
