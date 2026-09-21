"""Step-based continuation schedules compatible with existing LambdaLR state."""

import math

from torch.optim.lr_scheduler import LambdaLR


def linear_decay_continuation(
    optimizer, decay_start_step=6000, decay_end_step=10000,
    final_lr_ratio=0.1, last_epoch=-1):
  """Keep base LR until start, then linearly decay to a fraction of base LR.

  Steps are absolute optimizer-update counts, not updates since resume.
  Lightning restores optimizer state and LambdaLR.last_epoch from the full
  checkpoint. The new lambda remains configured here: loading the old
  Transformers constant-warmup LambdaLR state does not replace its function.
  This deliberately has no new warmup. Resume at decay_start_step to branch
  from the old constant-rate experiment without an initial LR discontinuity.
  """
  if (not isinstance(decay_start_step, int)
      or isinstance(decay_start_step, bool) or decay_start_step < 0):
    raise ValueError('decay_start_step must be a non-negative integer')
  if (not isinstance(decay_end_step, int)
      or isinstance(decay_end_step, bool)
      or decay_end_step <= decay_start_step):
    raise ValueError('decay_end_step must be an integer greater than start')
  if (isinstance(final_lr_ratio, bool)
      or not math.isfinite(final_lr_ratio) or not 0 <= final_lr_ratio <= 1):
    raise ValueError('final_lr_ratio must be finite and in [0, 1]')

  def multiplier(step):
    progress = min(1.0, max(0.0,
      (step - decay_start_step) / (decay_end_step - decay_start_step)))
    return 1.0 - (1.0 - final_lr_ratio) * progress

  return LambdaLR(optimizer, lr_lambda=multiplier, last_epoch=last_epoch)
