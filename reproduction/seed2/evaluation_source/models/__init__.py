import importlib

def __getattr__(name):
  """Load model modules only when selected.

  This keeps the exact structured utilities usable in a minimal Torch-only
  environment.  In particular, importing the coupling head does not require
  OmegaConf, Transformers, FlashAttention, or the Mamba kernels.
  """
  if name in {
      'dit', 'ema', 'crf_decoder', 'structured_decoder',
      'dimamba', 'autoregressive'}:
    module = importlib.import_module(f'{__name__}.{name}')
    globals()[name] = module
    return module
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
