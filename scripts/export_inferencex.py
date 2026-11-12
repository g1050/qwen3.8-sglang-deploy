#!/usr/bin/env python3
"""Stage archived AgentX aggregates as an InferenceX-app ingest bundle (CI mode layout).

The local dashboard keys this checkpoint as `qwen3.827b`; the L2 host cache that the runs
used is recorded as DRAM offloading, which the benchmark environment had left at `none`.
"""
import argparse
import json
from pathlib import Path

from deploy import ROOT, RUNTIME

MODEL_KEY = 'qwen3.827b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='runtime/agentx-c<N>-<stamp> directories')
    parser.add_argument('--output', type=Path, default=RUNTIME / 'inferencex-ingest')
    args = parser.parse_args()
    for run in args.runs:
        row = json.loads((run / 'aggregate.json').read_text())
        cache = json.loads((run / 'manifest.json').read_text())['deployment']['cache']
        row['infmax_model_prefix'] = MODEL_KEY
        if cache.get('l2_enabled'):
            row.update(kv_offloading='dram', kv_offload_backend='hicache',
                       allocated_cpu_dram_gb=cache['l2_total_gb'])
        name = f"agentic_{MODEL_KEY}_conc{row['conc']}_{row['precision']}_{row['framework']}_{run.name}"
        target = args.output / f'results_{name}'
        target.mkdir(parents=True, exist_ok=True)
        (target / f'{name}.json').write_text(json.dumps(row, indent=2) + '\n')
        print(target)


if __name__ == '__main__':
    main()
