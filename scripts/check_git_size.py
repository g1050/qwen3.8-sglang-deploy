#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
records = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--stage', '-z']).split(b'\0')
errors = []
for record in records:
    if not record:
        continue
    metadata, path = record.split(b'\t', 1)
    mode, oid, stage = metadata.split()
    if mode == b'160000':
        continue
    size = int(subprocess.check_output(['git', '-C', str(root), 'cat-file', '-s', oid]))
    if size > 1024 * 1024:
        errors.append(f'{path.decode()}: {size} bytes exceeds 1 MiB')
if errors:
    sys.exit('\n'.join(errors))
print('All staged files are at most 1 MiB')
