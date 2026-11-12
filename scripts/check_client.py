#!/usr/bin/env python3
"""Runs inside the client container: offline corpus load and upstream imports, no server access."""
import sys
import time

inferencex_e2e, dataset = sys.argv[1:3]
started = time.time()
from datasets import load_dataset

rows = load_dataset(dataset, split='train', trust_remote_code=False)
print('rows', len(rows), rows.column_names[:8], round(time.time() - started, 1), flush=True)
sys.path.insert(0, inferencex_e2e)
import infx.bench.agentic.replay  # noqa: E402,F401
import infx.results.agentic.analyze_benchmark_distributions  # noqa: E402,F401
import infx.results.agentic.process_agentic_result  # noqa: E402,F401
import infx.results.agentic.validate_agentic_result  # noqa: E402,F401

print('infx imports ok')
