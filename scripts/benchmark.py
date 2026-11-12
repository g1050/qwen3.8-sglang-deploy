#!/usr/bin/env python3
"""One AgentX point against the running performance container: warmup + measured window."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

from deploy import CFG, HOST, ROOT, RUNTIME, container_name, materialize, run

BENCH = CFG['benchmark']


def upstream():
    repo = Path(HOST['inferencex_dir'])
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != BENCH['inferencex_commit']:
        raise ValueError(f'InferenceX commit mismatch: {commit}')
    if subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--', 'inferencex-e2e/infx'], text=True):
        raise ValueError('InferenceX benchmark code has local modifications')
    return repo / 'inferencex-e2e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--duration', type=int, default=BENCH['duration'])
    options = parser.parse_args()
    e2e = upstream()
    deployment = materialize('performance')
    with urllib.request.urlopen(f"http://127.0.0.1:{CFG['port']}/server_info", timeout=30) as response:
        info = json.load(response)
    # The measured engine must be the simulated-acceptance MTP build of this project.
    if str(info.get('speculative_algorithm', '')).upper() != 'NEXTN' or info.get('speculative_num_steps') != CFG['performance']['args']['speculative-num-steps']:
        raise ValueError('Port is not served by the performance profile (native MTP)')
    snapshot = RUNTIME / 'hf/hub' / ('datasets--' + BENCH['dataset'].replace('/', '--')) / 'snapshots' / BENCH['dataset_revision']
    if not (snapshot / 'traces.jsonl').is_file():
        raise ValueError('Run scripts/prepare.py dataset first')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = RUNTIME / f'agentx-c{options.concurrency}-{stamp}'
    out.mkdir(parents=True)
    acceptance = CFG['performance']['acceptance']
    env = {
        'HF_HOME': str(RUNTIME / 'hf'), 'HF_HUB_CACHE': str(RUNTIME / 'hf/hub'),
        'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HOME': str(RUNTIME / 'client-home'), 'PYTHONUNBUFFERED': '1',
        'MODEL': CFG['model'], 'SERVED_MODEL_NAME': CFG['served_model_name'],
        'MODEL_PREFIX': CFG['model_prefix'], 'FRAMEWORK': 'sglang',
        'WEKA_LOADER_OVERRIDE': BENCH['dataset_loader'],
        'CONC': str(options.concurrency), 'DURATION': str(options.duration),
        'AIPERF_MAX_CONTEXT_LENGTH': str(BENCH['max_context_length']),
        'AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS': '3600', 'AIPERF_EXPERIMENTAL_FAST': '0',
        'AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID': 'false',
        'AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING': '0', 'AIPERF_HTTP_TCP_USER_TIMEOUT': '900000',
        'SRT_FRONTEND_HOST': '127.0.0.1', 'SRT_FRONTEND_PORT': str(CFG['port']),
        'AIPERF_SERVER_METRICS_URLS': f"http://127.0.0.1:{CFG['port']}/metrics",
        'TOKENIZER': HOST['model_dir'], 'RESULT_DIR': str(out), 'AGENTIC_OUTPUT_DIR': str(out),
        'RESULT_FILENAME': 'aggregate', 'IS_MULTINODE': 'false', 'PRECISION': 'fp8',
        'TP': '8', 'EP_SIZE': '1', 'PP_SIZE': '1', 'DCP_SIZE': '1', 'PCP_SIZE': '1', 'DP_ATTENTION': 'false',
        'KV_OFFLOADING': 'none', 'SPEC_DECODING': 'mtp', 'SPEC_ALGORITHM': 'nextn',
        'RUNNER_TYPE': 'rtx5090', 'IMAGE': CFG['image_upstream'], 'THINKING_MODE': acceptance['thinking_mode'],
    }
    plan = {'output': str(out), 'inferencex_e2e': str(e2e), 'agentx_version': BENCH['agentx_version'],
            'dataset_loader': BENCH['dataset_loader'], 'seed': BENCH['seed'],
            'failed_request_threshold': BENCH['failed_request_threshold'], 'env': env}
    manifest = dict(project_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                    started_at=stamp, concurrency=options.concurrency, duration=options.duration,
                    inferencex_commit=BENCH['inferencex_commit'], dataset_revision=BENCH['dataset_revision'],
                    dataset_sha256=BENCH['dataset_sha256'], dataset_bytes=BENCH['dataset_bytes'],
                    client_version=BENCH['agentx_version'], deployment=deployment,
                    server={k: info.get(k) for k in ('speculative_algorithm', 'speculative_num_steps', 'speculative_num_draft_tokens', 'tp_size', 'context_length', 'max_running_requests', 'mem_fraction_static', 'version')},
                    environment=env)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (RUNTIME / 'latest-run.txt').write_text(str(out) + '\n')
    (RUNTIME / 'client-home').mkdir(exist_ok=True)
    mounts = []
    for path in (ROOT, HOST['inferencex_dir'], HOST['dataset_dir'], HOST['model_dir']):
        mounts += ['-v', f'{path}:{path}' + ('' if path == ROOT else ':ro')]
    result = run(['docker', 'run', '--rm', '--name', f'qwen38-agentx-c{options.concurrency}', '--network=host',
                  '--ipc=host', '--gpus=all', *mounts, '-w', str(ROOT), '--entrypoint', str(RUNTIME / 'client/bin/python'),
                  CFG['image_id'], 'scripts/client_run.py', str(out / 'plan.json')], check=False)
    with (out / 'server.log').open('w') as stream:
        subprocess.run(['docker', 'logs', container_name('performance')], stdout=stream, stderr=subprocess.STDOUT)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
