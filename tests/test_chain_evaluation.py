"""Evaluation draw identities remain separate from local record indices."""
import json

import pytest

from scripts.evaluate_chain_crf import main


def arguments(output, offset=0):
    return ['--output', str(output), '--synthetic', '--device', 'cpu',
            '--length', '8', '--steps', '2', '--samples', '4',
            '--batch-size', '1', '--warmup', '0', '--sample-offset', str(offset)]


def read_records(output):
    return [json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]


def test_final_draw_set_is_disjoint_and_resume_preserves_it(tmp_path):
    screen, final = tmp_path/'screen', tmp_path/'final'
    main(arguments(screen))
    main(arguments(final, 10000))
    screen_rows, final_rows = read_records(screen), read_records(final)
    assert [r['sample_id'] for r in final_rows] == list(range(4))
    assert [r['draw_id'] for r in final_rows] == list(range(10000, 10004))
    assert not {r['draw_id'] for r in final_rows} & {r['draw_id'] for r in screen_rows}
    assert [r['token_ids'] for r in final_rows] != [r['token_ids'] for r in screen_rows]
    assert json.loads((final/'manifest.json').read_text())['config']['sample_offset'] == 10000
    (final/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in final_rows[:2]))
    main(arguments(final, 10000)+['--resume'])
    resumed = read_records(final)
    assert [(r['draw_id'], r['token_ids']) for r in resumed] == [
        (r['draw_id'], r['token_ids']) for r in final_rows]


def test_resume_rejects_changed_or_corrupt_draw_identity(tmp_path):
    output = tmp_path/'run'
    main(arguments(output, 100))
    with pytest.raises(ValueError, match='Resume manifest'):
        main(arguments(output, 200)+['--resume'])
    rows = read_records(output)
    rows[0]['draw_id'] = 0
    (output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='Stored draw IDs'):
        main(arguments(output, 100)+['--resume'])


def test_negative_offset_rejected_before_creating_output(tmp_path):
    output = tmp_path/'invalid'
    with pytest.raises(ValueError, match='nonnegative'):
        main(arguments(output, -1))
    assert not output.exists()


@pytest.mark.parametrize('mode,inference,device', [('backbone','dense','cpu'), ('count','dense','cpu'),
                                                ('count','segments','cpu'), ('count','dense','cuda')])
def test_profile_preserves_draws_and_excludes_warmup_and_later_batches(tmp_path,mode,inference,device):
    import torch
    from chain_crf.counts import CountBigramHead

    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA profiling requires a GPU')
    extra=['--mode',mode,'--inference',inference,'--device',device,'--warmup','1']
    if mode=='count':
        counts=tmp_path/'counts.pt'
        CountBigramHead(17).fit([[0,1,2,3,4]]).save(counts)
        extra+=['--counts',str(counts)]
    plain,profiled=tmp_path/'plain',tmp_path/'profiled'
    main(arguments(plain)+extra)
    main(arguments(profiled)+extra+['--profile'])
    rows=read_records(profiled)
    assert [r['token_ids'] for r in rows]==[r['token_ids'] for r in read_records(plain)]
    assert [r['profiled'] for r in rows]==[True,False,False,False]
    summary=json.loads((profiled/'profile-stages.json').read_text())
    stages={s['stage']:s for s in summary['stages']}
    assert len(stages)==len(summary['stages'])  # CUDA annotations must not duplicate regions.
    assert stages['mdlm.forward']['calls']==2  # Two steps, one batch; no warmup.
    if mode=='count':
        assert {'crf.candidates','crf.topk','crf.tail_mass','crf.potentials',
                'crf.forward_filter','crf.backward_sample','crf.residual_expand'}<=stages.keys()
    trace=json.loads((profiled/'profile.json').read_text())
    assert 'mdlm.forward' in {event.get('name') for event in trace['traceEvents']}
    assert summary['attribution']['method']=='cuda_launch_correlation_v1'
    assert summary['attribution']['missing_launch']['events']==0
    assert summary['attribution']['ambiguous_launch']['events']==0
    if device=='cuda' and mode=='count':
        fused=[e for e in trace['traceEvents']
               if e.get('cat')=='kernel' and e.get('name')=='_forward_kernel']
        assert fused  # Ensure this exercises actual driver-launched kernels.
        assert stages['crf.forward_filter']['device_kernel_ms']==pytest.approx(
            sum(e['dur'] for e in fused)/1000)
        assert stages['crf.inference']['device_kernel_ms']>=stages['crf.forward_filter']['device_kernel_ms']
        backward=[e for e in trace['traceEvents']
                  if e.get('cat')=='kernel' and e.get('name')=='_backward_sample_kernel']
        assert backward
        assert stages['crf.backward_fused']['device_kernel_ms']>=sum(e['dur'] for e in backward)/1000-1e-9
        assert stages['crf.backward_sample']['device_kernel_ms']>=stages['crf.backward_fused']['device_kernel_ms']
    assert 'Profiling adds overhead' in (profiled/'profile.txt').read_text()
    assert json.loads((profiled/'metrics.json').read_text())['contains_profiled_batches']
    assert not (plain/'profile.json').exists()


@pytest.mark.parametrize('flag', ['--score-only','--denoise-only'])
def test_profile_requires_generation(tmp_path,flag):
    output=tmp_path/'invalid'
    with pytest.raises(ValueError,match='requires generation'):
        main(arguments(output)+['--profile',flag])
    assert not output.exists()


@pytest.mark.parametrize('saved_count',[1,4])
def test_partial_batch_replays_original_draws_before_appending(tmp_path,saved_count):
    output=tmp_path/'batched'
    args=arguments(output,10000)
    args[args.index('--samples')+1]='7'
    args[args.index('--batch-size')+1]='3'
    main(args)
    expected=read_records(output)
    saved=expected[:saved_count]
    (output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in saved))
    main(args+['--resume'])
    resumed=read_records(output)
    keys=('sample_id','draw_id','token_ids','prefix_length','batch_id','batch_size')
    assert [[r[k] for k in keys] for r in resumed]==[[r[k] for k in keys] for r in expected]
    assert resumed[:saved_count]==saved


def test_partial_batch_refuses_changed_saved_prefix_before_appending(tmp_path):
    output=tmp_path/'tampered'
    args=arguments(output,10000)
    args[args.index('--batch-size')+1]='3'
    main(args)
    saved=read_records(output)[:1]
    saved[0]['token_ids'][0]=(saved[0]['token_ids'][0]+1)%16
    path=output/'samples.jsonl'
    path.write_text(json.dumps(saved[0])+'\n')
    before=path.read_bytes()
    with pytest.raises(ValueError,match='Replayed partial batch differs'):
        main(args+['--resume'])
    assert path.read_bytes()==before


def test_wall_time_excludes_loading_warmup_scoring_but_includes_sync_decode_io(tmp_path,monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    import scripts.evaluate_chain_crf as evaluator
    import chain_crf.generation as generation

    clock=SimpleNamespace(now=0.)
    def advance(seconds):
        clock.now+=seconds
    fake_time=SimpleNamespace(perf_counter=lambda: clock.now)
    monkeypatch.setattr(evaluator,'time',fake_time)
    monkeypatch.setattr(generation,'time',fake_time)
    monkeypatch.setattr(generation,'synchronize',lambda device: advance(3))
    original_model=evaluator.SyntheticBackbone
    def load_model(**kwargs):
        advance(1000)
        model=original_model(**kwargs)
        model.register_forward_hook(lambda *args: advance(2))
        def decode(row):
            advance(200)
            return ' '.join(map(str,row))
        model.tokenizer=SimpleNamespace(decode=decode)
        return model
    monkeypatch.setattr(evaluator,'SyntheticBackbone',load_model)
    def load_head(*args):
        advance(500)
        return None,{}
    monkeypatch.setattr(evaluator,'load_head',load_head)
    original_json=evaluator.atomic_json
    def write_json(data,path):
        advance(300 if path.name=='manifest.json' else 700)
        original_json(data,path)
    monkeypatch.setattr(evaluator,'atomic_json',write_json)
    def score(*args):
        advance(10000)
        return {'ppl':1.}
    monkeypatch.setattr(evaluator,'score_gpt2',score)
    original_open=Path.open
    class Stream:
        def __init__(self,stream): self.stream=stream
        def __enter__(self): return self
        def write(self,text):
            advance(5)
            return self.stream.write(text)
        def flush(self):
            advance(7)
            self.stream.flush()
        def __exit__(self,*args):
            self.stream.close()
            advance(13)
    def open_file(path,mode='r',*args,**kwargs):
        stream=original_open(path,mode,*args,**kwargs)
        return Stream(stream) if path.name=='samples.jsonl' and mode=='a' else stream
    monkeypatch.setattr(Path,'open',open_file)
    outputs=[]
    for warmup in (0,2):
        output=tmp_path/str(warmup)
        evaluator.main(arguments(output)+['--sampler','ddpm_cache','--warmup',str(warmup),
                                         '--samples','3','--batch-size','2','--score-gpt2'])
        metrics=json.loads((output/'metrics.json').read_text())
        rows=read_records(output)
        calls=sum(rows[i]['backbone_calls'] for i in (0,2))
        # Two production batches each have two explicit syncs (3s each).
        # Include manifest (300), decode (3*200), write/flush/close (55).
        expected=300+2*calls+12+600+55
        assert metrics['post_load_elapsed_seconds']==expected
        assert metrics['post_load_seconds_per_sample']==expected/3
        assert metrics['post_load_timing_complete']
        assert metrics['elapsed_seconds']<expected
        outputs.append(metrics['post_load_elapsed_seconds'])
    assert outputs[0]==outputs[1]


def test_wall_time_resume_does_not_invent_missing_session_time(tmp_path):
    output=tmp_path/'run'
    main(arguments(output))
    first=json.loads((output/'metrics.json').read_text())
    main(arguments(output)+['--resume'])
    same=json.loads((output/'metrics.json').read_text())
    assert same['post_load_elapsed_seconds']==first['post_load_elapsed_seconds']
    rows=read_records(output)
    (output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:2]))
    main(arguments(output)+['--resume'])
    resumed=json.loads((output/'metrics.json').read_text())
    assert resumed['post_load_elapsed_seconds'] is None
    assert resumed['post_load_seconds_per_sample'] is None
    assert not resumed['post_load_timing_complete']
