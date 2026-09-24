import argparse
from pathlib import Path
import sys
import torch
repo = Path('/u401/n23zhang/clean_tree_mdlm/mdlm-fork1')
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / 'experiments/original_table'))
from models.structured_decoder import SparseEdgeProposer
import run
original_forward = SparseEdgeProposer.forward

def no_chain_forward(self, node_context, active_mask, **kwargs):
    result = list(original_forward(self, node_context, active_mask, **kwargs))
    length = active_mask.shape[1]
    local_count = sum(length - offset for offset in range(1, min(self.local_window, length - 1) + 1))
    chain = slice(local_count, local_count + max(length - 1, 0))
    result[1] = result[1].clone()
    result[2] = result[2].clone()
    result[1][:, chain] = False
    result[2][:, chain] = -torch.inf
    return tuple(result)

SparseEdgeProposer.forward = no_chain_forward
p = argparse.ArgumentParser()
p.add_argument('--study', type=Path, required=True)
p.add_argument('--index', type=int, required=True)
a = p.parse_args()
run.evaluate(argparse.Namespace(study=a.study, suite='confirmation', index=a.index))
