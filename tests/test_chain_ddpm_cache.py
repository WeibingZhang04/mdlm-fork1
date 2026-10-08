"""CPU checks of native DDPM parity and custom-head reveal integration."""
import pytest
import torch

from chain_crf import ContextualPairHead, CountBigramHead, GlobalPairHead, IndependentHead
from chain_crf.generation import _generate_ddpm_cache
from test_native_sampling import ToyNative


class Backbone:
    vocab_size = 7
    mask_id = 6
    time_conditioning = False

    def __init__(self):
        self.history = []

    def __call__(self, tokens, t):
        self.history.append(tokens.clone())
        logits = torch.tensor([0., -.1, -.2, -.3, -.4, -.5, -torch.inf])
        return {'log_probs': logits.log_softmax(-1).expand(*tokens.shape, 7),
                'hidden': torch.zeros(*tokens.shape, 3)}


def run(backbone, tokens, steps=3, **kwargs):
    return _generate_ddpm_cache(backbone, tokens, steps,
        generator=torch.Generator().manual_seed(2735),
        reveal_generator=torch.Generator().manual_seed(1746), **kwargs)


@pytest.mark.parametrize('cap', [None, 1, 3, 500])
@pytest.mark.parametrize('eps', [1e-5, .4, .999999])
@pytest.mark.parametrize('final', [False, True])
@pytest.mark.parametrize('conditioned', [False, True])
def test_baseline_matches_actual_native_draws_and_call_counts(cap, eps, final, conditioned):
    native = ToyNative(cap=cap, noise_removal=final)
    native.time_conditioning = conditioned
    calls = []
    hook = native.backbone.register_forward_hook(lambda *args: calls.append(1))
    torch.manual_seed(2735)
    expected = native._sample(num_steps=3, eps=eps)
    hook.remove()
    backbone = Backbone()
    backbone.time_conditioning = conditioned
    actual, stats = run(backbone, torch.full((2, 12), 6),
                        vocab_cap=cap, eps=eps, noise_removal=final)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert stats['backbone_calls'] == len(calls) == len(backbone.history)
    assert stats['cache_hits'] == 3 + int(final) - len(calls)


def make_head(mode):
    torch.manual_seed(7)
    head = {'count': lambda: CountBigramHead(7).fit([[0, 1, 2, 3, 4, 5]] * 5),
            'global': lambda: GlobalPairHead(7, 2),
            'contextual': lambda: ContextualPairHead(7, 3, 2),
            'independent': lambda: IndependentHead(7, 3, 2)}[mode]()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0, .1)
    return head


@pytest.mark.parametrize('mode', ['count', 'global', 'contextual', 'independent'])
@pytest.mark.parametrize('inference,sampling', [('dense', 'joint'), ('segments', 'joint'),
                                             ('segments', 'marginal')])
@pytest.mark.parametrize('cap', [None, 2])
def test_heads_preserve_prefix_cap_and_reproducibility(mode, inference, sampling, cap):
    tokens = torch.full((2, 12), 6)
    tokens[:, :2] = torch.tensor([4, 5])
    head = make_head(mode)
    kwargs = dict(head=head, mode=mode, k=1, inference=inference,
                  sampling=sampling, vocab_cap=cap)
    actual, stats = run(Backbone(), tokens.clone(), **kwargs)
    again, _ = run(Backbone(), tokens.clone(), **kwargs)
    assert torch.equal(actual, again)
    assert torch.equal(actual[:, :2], tokens[:, :2])
    assert not actual.eq(6).any()
    if cap is not None:
        assert (actual[:, 2:] < cap).all()
    assert stats['backbone_calls'] >= 2  # Includes final backbone denoise.
    assert 0 <= stats['mean_retained_mass'] <= 1


def test_contextual_head_updates_time_while_reusing_cached_hidden_states():
    class RecordingHead(ContextualPairHead):
        def __init__(self):
            super().__init__(7, 3, 2)
            self.times, self.hidden_pointers = [], []

        def forward(self, ids, hidden, t):
            self.times.append(t.clone())
            self.hidden_pointers.append(hidden.data_ptr())
            return super().forward(ids, hidden, t)

    head = RecordingHead()
    actual, stats = run(Backbone(), torch.full((1, 4), 6), steps=4,
        head=head, mode='contextual', k=2, eps=.999999, noise_removal=False)
    assert actual.eq(6).all()
    assert stats['backbone_calls'] == 1 and stats['cache_hits'] == 3
    torch.testing.assert_close(torch.cat(head.times), torch.linspace(1, .999999, 5)[:-1])
    assert len(set(head.hidden_pointers)) == 1


def test_same_masked_backbone_probabilities_and_reveal_rng_give_same_reveals():
    histories = []
    for mode in ('count', 'global', 'contextual', 'independent'):
        backbone = Backbone()
        run(backbone, torch.full((2, 40), 6), steps=4, head=make_head(mode),
            mode=mode, k=2, inference='segments', eps=.2, noise_removal=False)
        histories.append(backbone.history)
    for history in histories[1:]:
        assert len(history) == len(histories[0])
        for actual, expected in zip(history, histories[0]):
            assert torch.equal(actual.eq(6), expected.eq(6))


def test_joint_crf_law_survives_independent_ddpm_reveals():
    class BinaryBackbone:
        mask_id = 2
        time_conditioning = False

        def __call__(self, tokens, t):
            return {'log_probs': torch.tensor([.5, .5, 0.]).log().expand(*tokens.shape, 3),
                    'hidden': torch.zeros(*tokens.shape, 1)}

    class AgreeHead(torch.nn.Module):
        def forward(self, ids, hidden, t):
            return 2. * ids[:, :-1, :, None].eq(ids[:, 1:, None, :]).float()

    samples, _ = run(BinaryBackbone(), torch.full((16000, 2), 2), steps=1,
        head=AgreeHead(), mode='global', k=2, eps=.4, noise_removal=False)
    empirical = torch.bincount(samples[:, 0] * 3 + samples[:, 1], minlength=9).float() / len(samples)
    expected = torch.zeros(3, 3)
    expected[:2, :2] = torch.tensor([2., 0., 0., 2.]).softmax(-1).reshape(2, 2) * .6**2
    expected[:2, 2] = expected[2, :2] = .5 * .6 * .4
    expected[2, 2] = .4**2
    torch.testing.assert_close(empirical, expected.flatten(), atol=.008, rtol=0)


@pytest.mark.parametrize('mode', ['count', 'global', 'contextual'])
def test_final_cleanup_uses_fresh_head_at_eps_and_keeps_visible_tokens(mode):
    class FavorOneHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.times = []

        def forward(self, ids, hidden, t):
            self.times.append(t.clone())
            bonus = 10. * ids.eq(1).float()
            return bonus[:, :-1, :, None] + bonus[:, 1:, None, :]

    head = FavorOneHead()
    tokens = torch.tensor([[5, 6, 6, 6]])
    actual, stats = run(Backbone(), tokens.clone(), steps=3, head=head,
        mode=mode, k=2, eps=.999999, noise_removal=True)
    assert actual.tolist() == [[5, 1, 1, 1]]  # Backbone alone would choose zero.
    assert len(head.times) == 4  # Three diffusion steps plus final head call.
    torch.testing.assert_close(head.times[-1], torch.tensor([.999999]))
    assert stats['backbone_calls'] == 2 and stats['cache_hits'] == 2
    assert stats['noise_removal_method'] == 'token_marginal_argmax'


def test_all_visible_custom_batch_has_finite_statistics():
    tokens = torch.tensor([[0, 1, 2]])
    actual, stats = run(Backbone(), tokens.clone(), head=make_head('global'),
        mode='global', k=2, inference='segments')
    assert torch.equal(actual, tokens)
    assert stats['mean_retained_mass'] == 1.


@pytest.mark.parametrize('mode', ['count', 'global', 'contextual'])
@pytest.mark.parametrize('cap', [None, 2])
@pytest.mark.parametrize('eps', [.4, .999999])
def test_custom_commit_uses_actual_native_reveals_and_crf_tokens(monkeypatch, mode, cap, eps):
    import chain_crf.generation as generation
    from test_native_sampling import NATIVE

    native = ToyNative(cap=cap, noise_removal=False)
    original_transition = generation._ddpm_transition
    original_draw = generation._draw_structured
    proposals, transitions = [], []

    def capture_draw(*args, **kwargs):
        drawn, mass = original_draw(*args, **kwargs)
        proposals.append(drawn.clone())
        return drawn, mass

    def checked_transition(p_x0, tokens, mask_id, t, dt, generator):
        # Independently obtain native clean probabilities for these inputs.
        native_lp = native.forward(tokens, t)
        native_p = native_lp.exp() if cap is None else NATIVE['capped_clean_probs'](
            native_lp, tokens, mask_id, cap)
        torch.testing.assert_close(p_x0, native_p, rtol=0, atol=0)
        # Run the actual native method from the same RNG state without
        # consuming or replacing the custom token-draw stream.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(generator.get_state())
            _, expected = native._ddpm_caching_update(tokens, t, dt, p_x0=native_p)
        actual = original_transition(p_x0, tokens, mask_id, t, dt, generator)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        transitions.append((tokens.clone(), expected.clone()))
        return actual

    monkeypatch.setattr(generation, '_draw_structured', capture_draw)
    monkeypatch.setattr(generation, '_ddpm_transition', checked_transition)
    tokens = torch.full((2, 12), 6)
    tokens[:, 0] = 5
    result, stats = run(Backbone(), tokens, steps=4, head=make_head(mode),
        mode=mode, k=1, inference='segments', vocab_cap=cap,
        eps=eps, noise_removal=False)
    assert len(transitions) == len(proposals) == 4
    for i, ((before, native_next), proposed) in enumerate(zip(transitions, proposals)):
        reveal = before.eq(6) & native_next.ne(6)
        expected_commit = torch.where(reveal, proposed, before)
        actual_commit = transitions[i + 1][0] if i + 1 < len(transitions) else result
        torch.testing.assert_close(actual_commit, expected_commit, rtol=0, atol=0)
    assert stats['reveal_sampler'] == 'native_categorical'


def enumerated_token_marginals(packet, unary, edge):
    """Enumerate tiny candidate chains, then distribute each tail's mass."""
    import itertools
    import math
    unary, edge = unary.detach(), edge.detach()
    b, length, states = unary.shape
    result = torch.zeros_like(packet.normalized_log_probs, dtype=torch.float64)
    for batch in range(b):
        total = 0.
        for path in itertools.product(range(states), repeat=length):
            score = sum(float(unary[batch, pos, state]) for pos, state in enumerate(path))
            score += sum(float(edge[batch, pos, path[pos], path[pos + 1]])
                         for pos in range(length - 1))
            if not math.isfinite(score):
                continue
            weight = math.exp(score)
            total += weight
            for pos, state in enumerate(path):
                token = int(packet.candidate_ids[batch, pos, state])
                if token >= 0:
                    result[batch, pos, token] += weight
                else:
                    conditional = packet.normalized_log_probs[batch, pos].double().exp()
                    explicit = packet.candidate_ids[batch, pos, :-1]
                    conditional[explicit] = 0.
                    conditional /= conditional.sum()
                    result[batch, pos] += weight * conditional
        result[batch] /= total
    return result


@pytest.mark.parametrize('mode', ['count', 'global', 'contextual', 'independent'])
@pytest.mark.parametrize('inference', ['dense', 'segments'])
@pytest.mark.parametrize('cap,k', [(None, 0), (None, 1), (None, 6), (2, 1), (2, 2), (2, 4)])
def test_final_token_argmax_matches_exhaustive_current_crf(mode, inference, cap, k):
    from chain_crf import build_candidates
    from chain_crf.generation import _denoise_structured, potentials

    tokens = torch.tensor([[5, 6, 6]])
    t = torch.tensor([1e-5])
    prediction = Backbone()(tokens, t)
    head = make_head(mode)
    packet = build_candidates(prediction['log_probs'], tokens, 6, k, vocab_cap=cap)
    unary, edge = potentials(packet, head, mode, prediction['hidden'], t)
    expected = enumerated_token_marginals(packet, unary, edge).argmax(-1)
    actual = _denoise_structured(prediction, tokens, 6, head, mode, t, k, inference, cap)
    assert torch.equal(actual, expected)
    assert actual[0, 0] == 5


@pytest.mark.parametrize('tail_multiplier,expected', [(1., 0), (2., 1)])
def test_final_argmax_compares_individual_tail_tokens_not_aggregate(tail_multiplier, expected):
    from chain_crf.generation import _denoise_structured

    class TailBias(torch.nn.Module):
        def forward(self, ids, hidden, t):
            return ids.eq(-1).float() * torch.tensor(tail_multiplier).log()

    tokens = torch.tensor([[4]])
    prediction = {'log_probs': torch.tensor([[[.4, .3, .2, .1, 0.]]]).log(),
                  'hidden': torch.zeros(1, 1, 3)}
    actual = _denoise_structured(prediction, tokens, 4, TailBias(), 'independent',
                                torch.tensor([1e-5]), 1, 'dense', None)
    # Initially tail mass=.6 exceeds the explicit token's .4, but every
    # individual tail token is smaller. Doubling tail mass makes token 1 win.
    assert actual.item() == expected


def test_final_argmax_respects_zero_tail_mass_and_token_id_ties():
    from chain_crf.generation import _denoise_structured

    tokens = torch.tensor([[3]])
    prediction = {'log_probs': torch.tensor([[[.5, .5, 0., 0.]]]).log(),
                  'hidden': torch.zeros(1, 1, 3)}
    result = _denoise_structured(prediction, tokens, 3, None, 'global',
                                torch.tensor([1e-5]), 2, 'dense', 2)
    assert result.item() == 0


@pytest.mark.parametrize('vocab', [7, 50258])
@pytest.mark.parametrize('scale', [1., 5., 20.])
def test_frozen_normalization_matches_vanilla_subs_in_fp32(vocab, scale):
    import ast
    from types import SimpleNamespace
    from chain_crf.backbone import FrozenMDLM
    from chain_crf.generation import _ddpm_clean_probs
    from test_native_sampling import ROOT

    class Encoder(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.register_buffer('logits', logits)

        def encode(self, tokens, conditioning):
            assert conditioning.eq(0).all()
            return torch.zeros(*tokens.shape, 3), conditioning

        def decode(self, hidden, conditioning):
            return self.logits.clone()

    raw_logits = torch.randn(2, 3, vocab, generator=torch.Generator().manual_seed(5)) * scale
    tokens = torch.full((2, 3), vocab - 1)
    tokens[:, 0] = 1
    # Exercise the real forward method without loading a checkpoint or tokenizer.
    backbone = FrozenMDLM.__new__(FrozenMDLM)
    torch.nn.Module.__init__(backbone)
    backbone.mask_id = vocab - 1
    backbone.encoder = Encoder(raw_logits)
    actual = backbone(tokens, torch.tensor([.8, .2]))
    assert actual['log_probs'].dtype == torch.float32
    actual_probs = _ddpm_clean_probs(actual['log_probs'], tokens, vocab - 1, None)

    # This original vanilla SUBS method predates the legacy CRF extension.
    tree = ast.parse((ROOT / 'diffusion.py').read_text())
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == '_subs_parameterization')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'vanilla_subs', 'exec'), namespace)
    native = SimpleNamespace(mask_index=vocab - 1, neg_infinity=-1000000.)
    expected = namespace['_subs_parameterization'](native, raw_logits.clone(), tokens)
    torch.testing.assert_close(actual_probs, expected.exp(), rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['backbone', 'global'])
@pytest.mark.parametrize('final', [False, True])
def test_timing_mode_preserves_draws_and_controls_explicit_synchronization(monkeypatch, mode, final):
    import chain_crf.generation as generation

    synchronizations = []
    monkeypatch.setattr(generation, 'synchronize', lambda device: synchronizations.append(str(device)))
    tokens = torch.full((2, 12), 6)
    kwargs = dict(mode=mode, head=None if mode == 'backbone' else make_head(mode),
                  k=2, noise_removal=final, eps=.999999)
    plain, timing = run(Backbone(), tokens.clone(), **kwargs)
    assert len(synchronizations) == 2  # Explicit start/end timing boundaries only.
    assert timing['timing_mode'] == 'outer_sync'
    assert timing['backbone_seconds'] is None and timing['sampling_seconds'] is None
    assert timing['elapsed_seconds'] >= 0
    synchronizations.clear()
    detailed, stages = run(Backbone(), tokens.clone(), stage_timing=True, **kwargs)
    assert len(synchronizations) == 2 + 2 * stages['backbone_calls'] + 3 + int(final)
    assert stages['timing_mode'] == 'stage_sync'
    assert stages['backbone_seconds'] >= 0 and stages['sampling_seconds'] >= 0
    assert stages['elapsed_seconds'] >= stages['backbone_seconds'] + stages['sampling_seconds']
    assert torch.equal(plain, detailed)
    assert timing['backbone_calls'] == stages['backbone_calls']


@pytest.mark.parametrize('mode', ['backbone', 'count', 'global', 'contextual', 'independent'])
def test_public_generate_dispatches_ddpm_with_prefix_and_default_cleanup(mode):
    from chain_crf.generation import generate

    kwargs = dict(sampler='ddpm_cache', length=8, steps=3, batch_size=2,
                  k=2, device='cpu', prefixes=[[4, 5], [5, 4]], sample_offset=17)
    head = None if mode == 'backbone' else make_head(mode)
    tokens, stats = generate(Backbone(), head, mode, **kwargs)
    assert tokens.shape == (2, 10)
    assert torch.equal(tokens[:, :2], torch.tensor([[4, 5], [5, 4]]))
    assert not tokens.eq(6).any()
    assert stats['sampler'] == 'ddpm_cache' and stats['noise_removal']
    assert stats['remaining_masks'] == 0 and stats['generated_tokens'] == 16
    assert stats['timing_mode'] == 'outer_sync' and stats['backbone_seconds'] is None
    again, _ = generate(Backbone(), head, mode, **kwargs)
    assert torch.equal(tokens, again)


@pytest.mark.parametrize('mode', ['backbone', 'count', 'global', 'contextual'])
def test_public_generate_keeps_fixed_as_default(mode):
    from chain_crf.generation import generate

    head = None if mode == 'backbone' else make_head(mode)
    kwargs = dict(length=8, steps=4, batch_size=2, k=2, device='cpu')
    default, stats = generate(Backbone(), head, mode, **kwargs)
    explicit, _ = generate(Backbone(), head, mode, sampler='fixed', **kwargs)
    assert torch.equal(default, explicit)
    assert stats['sampler'] == 'fixed' and not stats['noise_removal']
    assert stats['backbone_calls'] == 4 and stats['timing_mode'] == 'stage_sync'


@pytest.mark.parametrize('stage_timing', [False, True])
@pytest.mark.parametrize('mode', ['backbone', 'count'])
@pytest.mark.parametrize('cap', ['none', '2'])
def test_cli_ddpm_records_nullable_timing_and_replays_partial_batches(tmp_path, stage_timing, mode, cap):
    import json
    from scripts.evaluate_chain_crf import main

    output = tmp_path / 'run'
    args = ['--output', str(output), '--synthetic', '--device', 'cpu', '--mode', mode,
            '--sampler', 'ddpm_cache', '--steps', '2', '--length', '4', '--samples', '3',
            '--batch-size', '2', '--warmup', '1', '--k', '2', '--vocab-cap', cap]
    if mode == 'count':
        counts = tmp_path / 'counts.pt'
        CountBigramHead(17).fit([[0, 1, 2, 3, 4]]).save(counts)
        args += ['--counts', str(counts)]
    if stage_timing:
        args += ['--stage-timing']
    main(args)
    rows = [json.loads(line) for line in (output / 'samples.jsonl').read_text().splitlines()]
    metrics = json.loads((output / 'metrics.json').read_text())
    config = json.loads((output / 'manifest.json').read_text())['config']
    assert config['sampler'] == 'ddpm_cache' and config['noise_removal']
    assert config['stage_timing'] == stage_timing
    assert metrics['sampler'] == 'ddpm_cache'
    assert metrics['timing_mode'] == ('stage_sync' if stage_timing else 'outer_sync')
    assert (metrics['backbone_seconds'] is None) == (not stage_timing)
    assert metrics['elapsed_seconds'] == pytest.approx(sum(row['elapsed_seconds'] for row in rows))
    assert all(row['remaining_masks'] == 0 and len(row['token_ids']) == 4 for row in rows)
    assert metrics['noise_removal_method'] == ('backbone_argmax' if mode == 'backbone' else 'token_marginal_argmax')
    (output / 'samples.jsonl').write_text(json.dumps(rows[0]) + '\n')
    main(args + ['--resume'])
    resumed = [json.loads(line) for line in (output / 'samples.jsonl').read_text().splitlines()]
    assert [r['token_ids'] for r in resumed] == [r['token_ids'] for r in rows]
    assert resumed[0] == rows[0]
    with pytest.raises(ValueError, match='Resume manifest'):
        main(args + ['--resume', '--sampling-eps', '.1'])


@pytest.mark.parametrize('extra,match', [
    (['--sampler', 'ddpm_cache', '--temperature', '.5'], 'temperature'),
    (['--sampler', 'fixed', '--noise-removal'], 'requires --sampler'),
    (['--sampling-eps', '0'], 'sampling-eps'),
    (['--sampling-eps', '1'], 'sampling-eps'),
    (['--sampler', 'ddpm_cache', '--denoise-only'], 'requires generation'),
])
def test_invalid_sampler_options_fail_before_output_creation(tmp_path, extra, match):
    from scripts.evaluate_chain_crf import main

    output = tmp_path / 'bad'
    with pytest.raises(ValueError, match=match):
        main(['--output', str(output), '--synthetic', '--device', 'cpu'] + extra)
    assert not output.exists()


def test_cli_refuses_to_save_unremoved_masks(tmp_path):
    from scripts.evaluate_chain_crf import main

    output = tmp_path / 'masked'
    with pytest.raises(ValueError, match='enable --noise-removal'):
        main(['--output', str(output), '--synthetic', '--device', 'cpu',
              '--sampler', 'ddpm_cache', '--no-noise-removal', '--sampling-eps', '.999999',
              '--steps', '2', '--samples', '1', '--length', '4', '--warmup', '0'])
    assert not (output / 'samples.jsonl').exists()
