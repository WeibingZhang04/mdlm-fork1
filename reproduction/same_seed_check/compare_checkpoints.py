"""Compare checkpoint tensors without requiring serialization bytes to match."""
import json
import math
import sys
import torch

old, new = [torch.load(path, map_location='cpu') for path in sys.argv[1:3]]
assert old['global_step'] == new['global_step']
groups = {}
for name, x in old['state_dict'].items():
    y = new['state_dict'][name]
    assert x.shape == y.shape and x.dtype == y.dtype, name
    group = 'head' if name.startswith('structured_head.') else 'other_model_parameters'
    entry = groups.setdefault(group, dict(tensors=0, elements=0, different_elements=0, max_abs=0., squared_error=0.))
    entry['tensors'] += 1
    entry['elements'] += x.numel()
    if not torch.equal(x, y):
        delta = x.double() - y.double()
        entry['different_elements'] += int((x != y).sum())
        entry['max_abs'] = max(entry['max_abs'], float(delta.abs().max()))
        entry['squared_error'] += float(delta.square().sum())
for entry in groups.values():
    entry['rmse'] = math.sqrt(entry.pop('squared_error') / entry['elements'])
    entry['exactly_equal'] = entry['different_elements'] == 0
json.dump({'global_step': old['global_step'], 'original': sys.argv[1], 'retrained': sys.argv[2],
           'parameter_comparison': groups, 'scope': 'Model tensors only; serialization hashes need not match.'},
          open(sys.argv[3], 'w'), indent=2)
