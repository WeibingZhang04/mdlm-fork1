"""Exercise the actual native sampler with CPU fixtures, without model imports."""
import ast
import hashlib
import inspect
import itertools
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def sampler_methods(source):
    # Import only the real native sampling methods, avoiding Lightning,
    # FlashAttention and HF model loading in these CPU unit tests.
    tree = ast.parse(source)
    diffusion = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'Diffusion')
    names = {'_sample_prior','_ddpm_caching_update','_sample','restore_model_and_sample'}
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef)
                 and n.name in ('_sample_categorical','capped_clean_probs')]
    functions += [n for n in diffusion.body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace = {'torch':torch,'time':time,'itertools':itertools}
    exec(compile(ast.Module(body=functions,type_ignores=[]), 'diffusion.py', 'exec'),namespace)
    return namespace


def main_functions():
    tree = ast.parse((ROOT/'main.py').read_text())
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef)
                 and n.name in ('save_native_samples','generate_samples')]
    namespace = {'torch':torch,'Path':Path,'os':os,'hashlib':hashlib,'inspect':inspect,
                 'json':json,'platform':platform,'__file__':str(ROOT/'main.py')}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'main.py','exec'),namespace)
    return namespace


class ToyBackbone(torch.nn.Module):
    def forward(self, tokens, sigma):
        logits = torch.tensor([0.,-.1,-.2,-.3,-.4,-.5,-torch.inf])
        lp = logits.log_softmax(-1).expand(*tokens.shape,7).clone()
        visible = tokens.ne(6)
        lp[visible] = -torch.inf
        b,t = visible.nonzero(as_tuple=True)
        lp[b,t,tokens[visible]] = 0.
        return lp


class ToyNoise(torch.nn.Module):
    def forward(self,t):
        return t,torch.ones_like(t)


class ToyTokenizer:
    def batch_decode(self, tokens):
        return [' '.join(map(str,row)) for row in tokens.tolist()]


class ToyNative(torch.nn.Module):
    def __init__(self, output=None, cap=None, warmup=0, noise_removal=True):
        super().__init__()
        self.device = torch.device('cpu')
        self.mask_index = 6
        self.sampler = 'ddpm_cache'
        self.parameterization = 'subs'
        self.time_conditioning = False
        self.ema = None
        self.backbone = ToyBackbone()
        self.noise = ToyNoise()
        self.tokenizer = ToyTokenizer()
        self.config = Config(
            backbone='hf_dit', seed=1,
            sampling=Config(vocab_cap=cap, warmup_batches=warmup, steps=3,
                            num_sample_batches=2, noise_removal=noise_removal, semi_ar=False),
            loader=Config(eval_batch_size=2), model=Config(length=12),
            noise=Config(type='loglinear'),
            eval=Config(sample_output_dir=output, checkpoint_path='synthetic',disable_ema=False))
        self.gen_ppl_metric = SimpleNamespace(reset=lambda: None)

    def forward(self,tokens,sigma):
        return self.backbone(tokens,sigma)


NATIVE = sampler_methods((ROOT/'diffusion.py').read_text())
capped_clean_probs = NATIVE['capped_clean_probs']
MAIN = main_functions()
save_native_samples = MAIN['save_native_samples']
for name in ('_sample_prior','_ddpm_caching_update','_sample','restore_model_and_sample'):
    setattr(ToyNative,name,NATIVE[name])


@pytest.mark.parametrize('cap', [1,3,6,500])
def test_cap_normalizes_clean_unaries_and_keeps_observed_tokens(cap):
    tokens = torch.tensor([[6,5,6]])
    lp = ToyBackbone()(tokens,None).bfloat16()
    probs = capped_clean_probs(lp,tokens,6,cap)
    assert probs.dtype == torch.float32
    assert probs[0,1,5] == 1 and probs[0,1].sum() == 1
    assert probs[...,6].eq(0).all()
    torch.testing.assert_close(probs.sum(-1),torch.ones_like(tokens,dtype=torch.float32))
    k = min(cap,6)
    torch.testing.assert_close(probs[0,0,:k],lp[0,0,:k].float().softmax(-1))
    assert probs[0,0,k:].eq(0).all()


@pytest.mark.parametrize('cap', [1,3,6,500])
def test_ddpm_mask_mass_is_not_truncated_or_renormalized_away(monkeypatch,cap):
    model = ToyNative(cap=cap)
    tokens = torch.tensor([[6,5,6]])
    captured = []
    def draw(probs):
        captured.append(probs.clone())
        return probs.argmax(-1)
    monkeypatch.setitem(NATIVE,'_sample_categorical',draw)
    # Functions share NATIVE as their globals dictionary.
    _,result = model._ddpm_caching_update(tokens,torch.tensor([[.8]]),.2)
    q = captured[0]
    torch.testing.assert_close(q[0,[0,2],6],torch.tensor([.6,.6]))
    torch.testing.assert_close(q[0,[0,2],:6].sum(-1),torch.tensor([.2,.2]))
    torch.testing.assert_close(q[0,[0,2],6]/q[0,[0,2]].sum(-1),torch.tensor([.75,.75]))
    assert result[0,1] == 5


def test_native_cached_update_reuses_predictions():
    model = ToyNative(cap=2)
    tokens = torch.full((1,4),6)
    lp = model.forward(tokens,torch.tensor([1.]))
    cached = capped_clean_probs(lp,tokens,6,2)
    def forbidden(*args):
        raise AssertionError('cache hit called the backbone')
    model.forward = forbidden
    returned,_ = model._ddpm_caching_update(tokens,torch.tensor([[.5]]),.1,p_x0=cached)
    assert returned is cached


def test_final_denoise_respects_cap_even_with_tied_top_tokens(monkeypatch):
    model = ToyNative(cap=1)
    def tied_forward(tokens, sigma):
        scores = torch.zeros(*tokens.shape,7)
        scores[...,6] = -torch.inf
        return scores.log_softmax(-1)
    model.forward = tied_forward
    # Keep every position masked so that final noise removal must fill it.
    monkeypatch.setitem(NATIVE,'_sample_categorical',lambda q: torch.full(q.shape[:-1],6))
    expected = capped_clean_probs(tied_forward(torch.full((2,12),6),None),
                                 torch.full((2,12),6),6,1).argmax(-1)
    torch.testing.assert_close(model._sample(),expected)


@pytest.mark.parametrize('cap', [0,-1,'none',2.5,True])
def test_invalid_native_cap_rejected(cap):
    with pytest.raises(ValueError,match='positive integer'):
        ToyNative(cap=cap)._sample()


def test_unsupported_native_sampler_rejects_cap():
    model = ToyNative(cap=2)
    model.sampler = 'ddpm'
    with pytest.raises(ValueError,match='ddpm_cache'):
        model._sample()


@pytest.mark.parametrize('cap', [None,2])
def test_export_excludes_warmup_preserves_rng_and_records_real_calls(tmp_path,cap):
    outputs = []
    for warmup in (0,2):
        path = tmp_path/str(warmup)
        model = ToyNative(output=str(path),cap=cap,warmup=warmup)
        all_calls = []
        handle = model.backbone.register_forward_hook(lambda *args: all_calls.append(1))
        torch.manual_seed(1)
        save_native_samples(model,model.config)
        handle.remove()
        records = [json.loads(row) for row in (path/'samples.jsonl').read_text().splitlines()]
        metrics = json.loads((path/'metrics.json').read_text())
        manifest = json.loads((path/'manifest.json').read_text())
        outputs.append([r['token_ids'] for r in records])
        assert len(records) == 4
        assert metrics['elapsed_seconds'] == pytest.approx(sum(r['elapsed_seconds'] for r in records))
        assert metrics['seconds_per_sample'] == pytest.approx(metrics['elapsed_seconds']/4)
        batch_calls = sum(records[i]['backbone_calls'] for i in (0,2))
        assert metrics['actual_backbone_calls'] == batch_calls
        assert len(all_calls) == batch_calls if warmup == 0 else len(all_calls) > batch_calls
        assert manifest['warmup_batches'] == warmup and manifest['vocab_cap'] == cap
        assert not model.backbone._forward_hooks
        assert all(6 not in row for row in outputs[-1])
        if cap:
            assert all(token < cap for row in outputs[-1] for token in row)
    assert outputs[0] == outputs[1]


def test_final_denoise_adds_one_measured_backbone_call(tmp_path):
    counts = []
    for final in (False,True):
        model = ToyNative(output=str(tmp_path/str(final)),noise_removal=final)
        torch.manual_seed(1)
        save_native_samples(model,model.config)
        metrics = json.loads((Path(model.config.eval.sample_output_dir)/'metrics.json').read_text())
        counts.append(metrics['actual_backbone_calls'])
    assert counts[1] == counts[0]+2  # Two measured batches, one extra call each.


def test_export_refuses_overwrite(tmp_path):
    (tmp_path/'samples.jsonl').write_text('existing')
    model = ToyNative(output=str(tmp_path))
    with pytest.raises(FileExistsError):
        save_native_samples(model,model.config)
    assert (tmp_path/'samples.jsonl').read_text() == 'existing'


def test_native_seconds_exclude_loading_setup_warmup_decoding_and_output(tmp_path,monkeypatch):
    clock = SimpleNamespace(now=0.)
    def advance(seconds):
        clock.now += seconds
    monkeypatch.setitem(NATIVE,'time',SimpleNamespace(perf_counter=lambda: clock.now))
    model = ToyNative(output=str(tmp_path),warmup=2)

    def load_model(**kwargs):
        advance(1000.)  # Model loading happens before entering the sampler.
        return model
    monkeypatch.setitem(MAIN,'_load_from_checkpoint',load_model)
    original_prior = model._sample_prior
    def slow_setup(*args):
        advance(100.)  # Per-batch token setup is also outside the timer.
        return original_prior(*args)
    model._sample_prior = slow_setup
    handle = model.backbone.register_forward_hook(lambda *args: advance(2.))
    original_decode = model.tokenizer.batch_decode
    def slow_decode(tokens):
        advance(200.)
        return original_decode(tokens)
    model.tokenizer.batch_decode = slow_decode
    original_write = Path.write_text
    def slow_write(path,*args,**kwargs):
        advance(300.)
        return original_write(path,*args,**kwargs)
    monkeypatch.setattr(Path,'write_text',slow_write)
    try:
        MAIN['generate_samples'](model.config,SimpleNamespace(info=lambda text: None),model.tokenizer)
    finally:
        handle.remove()
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    # Only measured forwards contribute: none of the fake costs above, and
    # none of the forwards performed during the two warmup batches.
    assert metrics['elapsed_seconds'] == 2.*metrics['actual_backbone_calls']
    assert metrics['seconds_per_sample'] == metrics['elapsed_seconds']/4
    assert clock.now > metrics['elapsed_seconds']+1000.
    assert not (tmp_path/'gpt2-large.json').exists()  # Scoring is a later command.


def test_native_export_has_no_dependency_on_removed_module(tmp_path):
    assert not (ROOT/'native_sampling.py').exists()
    for name in ('main.py','diffusion.py'):
        assert 'from native_sampling import' not in (ROOT/name).read_text()
    model = ToyNative(output=str(tmp_path))
    save_native_samples(model,model.config)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    assert str(ROOT/'main.py') in manifest['source_sha256']
    assert str(ROOT/'diffusion.py') in manifest['source_sha256']
    assert not any(Path(path).name=='native_sampling.py' for path in manifest['source_sha256'])
