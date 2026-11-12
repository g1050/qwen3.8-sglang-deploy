#!/usr/bin/env python3
"""Prepare the benchmark client and the offline dataset cache (domestic mirrors by default)."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from deploy import CFG, HOST, ROOT, RUNTIME, run, sha256_file

BENCH = CFG['benchmark']
HF_HOME = RUNTIME / 'hf'
MIRROR = HOST.get('pip_index_url', 'https://pypi.tuna.tsinghua.edu.cn/simple')


def in_image(script):
    """Python 3.12 of the pinned SGLang image builds the client; the host only has 3.10."""
    run(['docker', 'run', '--rm', '--network=host', '-v', f'{ROOT}:{ROOT}', '-w', str(ROOT),
         '--entrypoint=bash', CFG['image_id'], '-ec', script])


def client():
    in_image(f'''PY=$(ls /usr/bin/python3.1[1-3] | head -1)
$PY -m venv --clear runtime/client
runtime/client/bin/python -m pip install -q -i {MIRROR} uv
runtime/client/bin/uv pip sync --python runtime/client/bin/python --require-hashes --index-url {MIRROR} config/client.lock
runtime/client/bin/python -c "from importlib.metadata import version; assert version('agentx-harness') == '{BENCH['agentx_version']}'"''')


def lock():
    in_image(f'''runtime/client/bin/uv pip compile --python runtime/client/bin/python --generate-hashes --index-url {MIRROR} config/client.in -o config/client.lock -q
sed -i "/^--index-url/d" config/client.lock''')


def dataset():
    """Expose the already-downloaded corpus as a pinned, offline Hugging Face hub cache."""
    source = Path(HOST['dataset_dir'])
    traces = source / 'traces.jsonl'
    if not traces.is_file():
        endpoint = HOST.get('hf_endpoint', 'https://hf-mirror.com')
        run([RUNTIME / 'client/bin/hf', 'download', BENCH['dataset'], '--repo-type', 'dataset',
             '--revision', BENCH['dataset_revision'], '--local-dir', source],
            env={**os.environ, 'HF_ENDPOINT': endpoint})
    if any(source.glob('*.incomplete')) or traces.stat().st_size != BENCH['dataset_bytes']:
        raise ValueError('Dataset download is incomplete')
    if sha256_file(traces) != BENCH['dataset_sha256']:
        raise ValueError('Dataset content hash does not match the recorded corpus')
    repo = HF_HOME / 'hub' / ('datasets--' + BENCH['dataset'].replace('/', '--'))
    snapshot = repo / 'snapshots' / BENCH['dataset_revision']
    snapshot.mkdir(parents=True, exist_ok=True)
    (repo / 'refs').mkdir(exist_ok=True)
    (repo / 'refs/main').write_text(BENCH['dataset_revision'])
    for name in ('traces.jsonl', 'README.md'):
        link = snapshot / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source / name)
    # One metadata-only resolution through the mirror builds the datasets cache from the local
    # file; every benchmark afterwards runs with HF_HUB_OFFLINE=1.
    (RUNTIME / 'client-home').mkdir(exist_ok=True)
    mounts = []
    for path in (ROOT, HOST['inferencex_dir'], source):
        mounts += ['-v', f'{path}:{path}' + ('' if path == ROOT else ':ro')]
    run(['docker', 'run', '--rm', '--network=host', *mounts, '-w', str(ROOT),
         '-e', f'HF_HOME={HF_HOME}', '-e', f"HF_HUB_CACHE={HF_HOME / 'hub'}", '-e', f"HOME={RUNTIME / 'client-home'}",
         '-e', 'HF_ENDPOINT=' + HOST.get('hf_endpoint', 'https://hf-mirror.com'),
         '--entrypoint', str(RUNTIME / 'client/bin/python'), CFG['image_id'], 'scripts/check_client.py',
         str(Path(HOST['inferencex_dir']) / 'inferencex-e2e'), BENCH['dataset']])
    if (repo / 'refs/main').read_text().strip() != BENCH['dataset_revision'] or len(list((repo / 'snapshots').iterdir())) != 1:
        raise ValueError('Mirror resolved a different dataset revision than the recorded corpus')
    print(json.dumps({'snapshot': str(snapshot), 'sha256': BENCH['dataset_sha256'], 'bytes': BENCH['dataset_bytes']}))


def source():
    target = Path(HOST['inferencex_dir'])
    if target.exists():
        raise ValueError('Source directory exists; verify its revision without overwriting it')
    subprocess.run(['git', 'clone', '--no-checkout', BENCH['inferencex_url'], str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', BENCH['inferencex_commit']], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['client', 'lock', 'dataset', 'source'])
    {'client': client, 'lock': lock, 'dataset': dataset, 'source': source}[parser.parse_args().action]()
