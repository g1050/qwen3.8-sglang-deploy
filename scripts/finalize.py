#!/usr/bin/env python3
"""Copy the small summaries of a run into results/ for version control."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from deploy import ROOT, RUNTIME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', type=Path)
    args = parser.parse_args()
    run = args.run or Path((RUNTIME / 'latest-run.txt').read_text().strip())
    if not (run / 'exit-codes.json').is_file():
        raise ValueError('Benchmark or result processing has not finished')
    report = subprocess.run([sys.executable, str(ROOT / 'scripts/report.py'), str(run)])
    destination = ROOT / 'results' / run.name
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('summary.json', 'summary.md', 'exit-codes.json', 'manifest.json', 'command.txt', 'aggregate.json',
                 'aiperf_artifacts/profile_export_aiperf.json', 'aiperf_artifacts/profile_export_console.txt'):
        source = run / name
        if source.is_file() and source.stat().st_size <= 1024 * 1024:
            shutil.copy2(source, destination / Path(name).name)
    validation = run / 'validate.log'
    if validation.is_file() and validation.stat().st_size <= 1024 * 1024:
        shutil.copy2(validation, destination / 'validation.txt')
    print(f'Small result summaries saved to {destination}')
    sys.exit(report.returncode)


if __name__ == '__main__':
    main()
