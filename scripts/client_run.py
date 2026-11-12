#!/usr/bin/env python3
"""Runs inside the client container: upstream ReplayConfig/replay_argv, AgentX, upstream post-processing."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    out = Path(plan['output'])
    sys.path.insert(0, plan['inferencex_e2e'])
    from importlib.metadata import version
    from infx.bench.agentic.replay import ReplayConfig, replay_argv

    if version('agentx-harness') != plan['agentx_version']:
        raise ValueError('AgentX version mismatch')
    env = {**os.environ, **plan['env'], 'PYTHONPATH': plan['inferencex_e2e']}
    cfg = ReplayConfig.from_env(env, out)
    if cfg.loader != plan['dataset_loader'] or cfg.unsafe_override:
        raise ValueError('Replay does not use the pinned full corpus and standard scenario')
    command = replay_argv(cfg, Path(sys.executable).with_name('agentx')) + ['--random-seed', str(plan['seed'])]
    (out / 'command.txt').write_text(shlex.join(command) + '\n')
    print(shlex.join(command), flush=True)
    with (out / 'benchmark.log').open('w') as stream:
        checks = {'benchmark': subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode}
    modules = [
        ('aggregate', 'infx.results.agentic.process_agentic_result', []),
        ('validate', 'infx.results.agentic.validate_agentic_result',
         [str(cfg.artifact_dir), '--failed-request-threshold', plan['failed_request_threshold']]),
        ('distributions', 'infx.results.agentic.analyze_benchmark_distributions', [str(cfg.artifact_dir), '-o', str(out)]),
    ]
    for name, module, args in modules:
        with (out / f'{name}.log').open('w') as stream:
            checks[name] = subprocess.run([sys.executable, '-m', module, *args], env=env,
                                          stdout=stream, stderr=subprocess.STDOUT).returncode
    metrics = cfg.artifact_dir / 'server_metrics_export.json'
    csv = cfg.artifact_dir / 'server_metrics_export.csv'
    checks['server_metrics'] = 0 if metrics.is_file() and csv.is_file() and csv.stat().st_size > 0 and 'sglang:' in metrics.read_text() else 1
    (out / 'exit-codes.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'checks': checks}), flush=True)
    sys.exit(1 if any(checks.values()) else 0)


if __name__ == '__main__':
    main()
