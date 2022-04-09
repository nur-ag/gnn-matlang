import os
import sys
import glob
from collections import deque


def fetch_last_lines(file_path, n=4):
    with open(file_path, 'r') as f:
        return list(deque(f, maxlen=n))

def check_lines_pattern(last_lines):
    if len(last_lines) != 4:
        return False

    # Last epoch of the last split
    first = last_lines[0]
    remaining = last_lines[1:]
    if not first.startswith('09 Epoch:'):
        return False

    # This script doesn't print final epoch scores!
    second = remaining[0]
    if second.startswith('09 Epoch:'):
        remaining = remaining[1:]

    # Two or three 'score' lines -- for the epoch and complete
    for line in remaining:
        try:
            value = float(line.strip())
        except ValueError:
            return False
    return True

# Store valid and invalid files
target_path = sys.argv[1] if len(sys.argv) > 1 else '.'
dry_run = (sys.argv[2] != 'remove') if len(sys.argv) > 2 else False
print(f'Looking for results under "{target_path}/". Dry-run: {dry_run}.')

valid_files = []
invalid_files = []
for file_name in glob.glob(f'{target_path}/*.txt'):
    last_lines = fetch_last_lines(file_name)
    target_list = invalid_files
    if check_lines_pattern(last_lines):
        target_list = valid_files
    target_list.append(file_name)


print(f'Found {len(valid_files)} valid and {len(invalid_files)} invalid result files.')

# Delete invalid files or print their names
for file in invalid_files:
    if dry_run:
        print(file)
    else:
        os.remove(file)

