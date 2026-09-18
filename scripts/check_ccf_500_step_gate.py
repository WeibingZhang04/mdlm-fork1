"""Require all eight configuration-specific checks before generation."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]) / 'verification'
for index in range(8):
    reports = list(root.glob(f'{index:02d}_*/attempt-*/verification.json'))
    assert len(reports) == 1, (index, reports)
    report = json.loads(reports[0].read_text())
    assert report['status'] == 'passed' and all(report['checks'].values()), report
print('All eight 500-step configuration equivalence gates passed.')
