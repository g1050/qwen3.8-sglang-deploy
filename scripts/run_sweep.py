#!/usr/bin/env python3
"""Run every configured concurrency on a fresh performance server, then start the real service.

Each point gets a new container so no prefix cache carries over. Whatever happens, the
performance container is removed and the default service (TTL and L3 off) is started.
"""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys

from deploy import CFG, ROOT, RUNTIME

STATE = RUNTIME / 'sweep-state.json'


def state(stage, **details):
    value = {'stage': stage, 'updated_at': datetime.now(timezone.utc).isoformat(), **details}
    STATE.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def step(stage, argv, check=True):
    state(stage)
    with (RUNTIME / f'sweep-{stage}.log').open('w') as stream:
        code = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    if code and check:
        raise RuntimeError(f'{stage} exited {code}')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--concurrency', type=int, nargs='+', default=CFG['benchmark']['concurrency'])
    parser.add_argument('--duration', type=int, default=CFG['benchmark']['duration'])
    parser.add_argument('--no-service', action='store_true', help='Do not start the default service afterwards')
    options = parser.parse_args()
    deploy = [sys.executable, 'scripts/deploy.py']
    perf = deploy + ['--profile', 'performance']
    results, error = {}, None
    try:
        step('stop-service', deploy + ['stop'], check=False)
        for conc in options.concurrency:
            tag = f'c{conc}'
            step(f'{tag}-clean', perf + ['stop'], check=False)
            step(f'{tag}-remove', perf + ['remove'], check=False)
            step(f'{tag}-start', perf + ['start'])
            step(f'{tag}-wait', perf + ['wait'])
            code = step(f'{tag}-benchmark', [sys.executable, 'scripts/benchmark.py', '--concurrency', str(conc),
                                             '--duration', str(options.duration)], check=False)
            run = (RUNTIME / 'latest-run.txt').read_text().strip()
            results[tag] = {'run': run, 'benchmark_exit': code,
                            'archive_exit': step(f'{tag}-archive', [sys.executable, 'scripts/finalize.py', run], check=False)}
    except BaseException as caught:
        error = caught
    finally:
        step('final-stop-performance', perf + ['stop'], check=False)
        step('final-remove-performance', perf + ['remove'], check=False)
        if not options.no_service:
            if step('service-start', deploy + ['start'], check=False):
                # The stopped service container of this project already exists.
                step('service-restart', ['docker', 'start', CFG['container_name']])
            step('service-wait', deploy + ['wait'])
    if error:
        state('failed-service-restored', error_type=type(error).__name__, error=str(error), results=results)
        raise error
    state('complete-service-restored', results=results)


if __name__ == '__main__':
    main()
