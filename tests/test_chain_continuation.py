"""Exact-prefix transfer generation, source identities, scoring and recovery."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from chain_crf import GlobalPairHead
from chain_crf.generation import generate
import scripts.evaluate_chain_crf as evaluator


class RecordingBackbone:
    vocab_size=5
    mask_id=4

    def __init__(self):
        self.history=[]

    def __call__(self,tokens,time):
        self.history.append(tokens.clone())
        logits=torch.tensor([0.,-.4,-.8,-1.1,-torch.inf],device=tokens.device)
        hidden=torch.nn.functional.one_hot(tokens.remainder(3),num_classes=3).float()
        return {'log_probs':logits.log_softmax(-1).expand(*tokens.shape,self.vocab_size),
                'hidden':hidden}


def source_rows(shapes=((2,4),(2,4),(2,4),(3,2),(3,2),(1,5))):
    rows=[]
    for index,(prefix_length,suffix_length) in enumerate(shapes):
        rows.append({'prefix_input_ids':[(index+j)%16 for j in range(prefix_length)],
                     'reference_continuation_ids':[(index+j+4)%16 for j in range(suffix_length)],
                     'prefix_length':prefix_length,'document_id':f'article-{index//2}',
                     'chunk_id':f'chunk-{index}','source_split':'validation',
                     'source_start_row':index*2,'source_stop_row':index*2+2,
                     'token_start':0,'token_stop':prefix_length+suffix_length})
    return rows


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def records(output):
    return [json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]


def arguments(output,source,count=6):
    return ['--output',str(output),'--synthetic','--device','cpu','--steps','2',
            '--samples',str(count),'--batch-size','3','--warmup','0',
            '--sample-offset','10000','--continuation-file',str(source)]


@pytest.mark.parametrize('mode,inference',[('backbone','dense'),('global','dense'),('global','segments')])
def test_distinct_exact_prefixes_remain_clamped_every_step(mode,inference):
    backbone=RecordingBackbone()
    head=GlobalPairHead(5,rank=2) if mode=='global' else None
    prefixes=[[0,1],[2,3]]
    tokens,stats=generate(backbone,head,mode,length=5,steps=3,batch_size=2,
        device='cpu',prefixes=prefixes,k=3,sample_offset=40,inference=inference)
    assert tokens.shape==(2,7)
    assert tokens[:,:2].tolist()==prefixes
    assert all(value[:,:2].tolist()==prefixes for value in backbone.history)
    assert stats['generated_tokens']==10 and stats['prefix_length']==2


@pytest.mark.parametrize('prefix,error',[
    ([4],'absorbing masks'),([-1],'out-of-vocabulary'),([5],'out-of-vocabulary'),
    ([True],'integer'),([1.0],'integer')])
def test_invalid_prefix_rejected_before_backbone_call(prefix,error):
    backbone=RecordingBackbone()
    with pytest.raises(ValueError,match=error):
        generate(backbone,length=3,steps=2,batch_size=1,device='cpu',prefixes=[prefix])
    assert not backbone.history


def test_shared_prefix_and_same_exact_prefix_batch_have_identical_draws():
    for prefix in ([],[2,1]):
        kwargs=dict(length=7,steps=3,batch_size=3,device='cpu',sample_offset=17)
        shared,_=generate(RecordingBackbone(),prefix=prefix,**kwargs)
        individual,_=generate(RecordingBackbone(),prefixes=[prefix]*3,**kwargs)
        torch.testing.assert_close(shared,individual,rtol=0,atol=0)
    with pytest.raises(ValueError,match='equal lengths'):
        generate(RecordingBackbone(),prefixes=[[0],[0,1]],batch_size=2,device='cpu')
    with pytest.raises(ValueError,match='not both'):
        generate(RecordingBackbone(),prefix=[0],prefixes=[[0]],device='cpu')


def test_loader_selection_and_shape_batches_preserve_source_order(tmp_path):
    path=write_rows(tmp_path/'input.jsonl',source_rows())
    rows,checksum=evaluator.load_continuations(path,vocab_size=17,mask_id=16)
    assert checksum==hashlib.sha256(path.read_bytes()).hexdigest()
    assert evaluator.continuation_batch_plan(rows,3)==[(0,3),(3,2),(5,1)]
    selected=evaluator.select_continuations(rows,offset=1,samples=2,one_per_document=True)
    assert [row['input_index'] for row in selected]==[2,4]
    with pytest.raises(ValueError,match='exceeds'):
        evaluator.select_continuations(rows,offset=0,samples=4,one_per_document=True)


@pytest.mark.parametrize('change,error',[
    (lambda rows:rows[0].update(prefix_input_ids=[16,1]),'absorbing masks'),
    (lambda rows:rows[0].update(prefix_input_ids=[17,1]),'out-of-vocabulary'),
    (lambda rows:rows[0].update(reference_continuation_ids=[]),'nonempty'),
    (lambda rows:rows[0].update(prefix_length=20),'prefix_length'),
    (lambda rows:rows[0].update(token_stop=100),'token offsets'),
    (lambda rows:rows[1].update(chunk_id='chunk-0'),'Duplicate'),
    (lambda rows:rows[1].update(source_split='test'),'Do not mix'),
    (lambda rows:rows[0].update(document_id=''),'document_id'),
])
def test_invalid_collection_rejected(tmp_path,change,error):
    rows=source_rows()
    change(rows)
    path=write_rows(tmp_path/'bad.jsonl',rows)
    with pytest.raises(ValueError,match=error):
        evaluator.load_continuations(path,vocab_size=17,mask_id=16)


def test_cli_uses_exact_ids_variable_lengths_and_never_tokenizes(tmp_path,monkeypatch):
    original=evaluator.SyntheticBackbone

    class NoEncodingTokenizer:
        def encode(self,*args,**kwargs):
            raise AssertionError('Exact-prefix mode must never tokenize text')
        def decode(self,ids):
            return 'display only'

    def with_tokenizer(**kwargs):
        model=original(**kwargs)
        model.tokenizer=NoEncodingTokenizer()
        return model

    monkeypatch.setattr(evaluator,'SyntheticBackbone',with_tokenizer)
    inputs=source_rows()
    source=write_rows(tmp_path/'source.jsonl',inputs)
    output=tmp_path/'run'
    evaluator.main(arguments(output,source))
    actual=records(output)
    assert [row['draw_id'] for row in actual]==list(range(10000,10006))
    for index,(row,expected) in enumerate(zip(actual,inputs)):
        assert row['input_index']==index
        assert row['document_id']==expected['document_id']
        assert row['chunk_id']==expected['chunk_id']
        assert row['token_ids'][:row['prefix_length']]==expected['prefix_input_ids']
        assert len(row['token_ids'])==expected['token_stop']
        assert row['generated_length']==len(expected['reference_continuation_ids'])
        assert row['continuation_file_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['continuations']['file_sha256']==actual[0]['continuation_file_sha256']
    metrics=json.loads((output/'metrics.json').read_text())
    expected_tokens=sum(len(row['reference_continuation_ids']) for row in inputs)
    assert metrics['tokens']==expected_tokens
    assert metrics['generated_tokens_per_second']==pytest.approx(expected_tokens/metrics['elapsed_seconds'])


def test_reference_values_do_not_influence_generation(tmp_path):
    first=source_rows()
    second=[{**row,'reference_continuation_ids':[(v+3)%16 for v in row['reference_continuation_ids']]}
            for row in first]
    outputs=[]
    for name,rows in [('first',first),('second',second)]:
        path=write_rows(tmp_path/f'{name}.jsonl',rows)
        output=tmp_path/name
        evaluator.main(arguments(output,path))
        outputs.append(records(output))
    assert [row['token_ids'] for row in outputs[0]]==[row['token_ids'] for row in outputs[1]]


def test_cli_article_selection_offset_is_independent_of_global_draw_ids(tmp_path):
    source=write_rows(tmp_path/'source.jsonl',source_rows())
    output=tmp_path/'selected'
    evaluator.main(arguments(output,source,count=2)+[
        '--one-per-document','--continuation-offset','1'])
    actual=records(output)
    assert [row['input_index'] for row in actual]==[2,4]
    assert [row['chunk_id'] for row in actual]==['chunk-2','chunk-4']
    assert [row['draw_id'] for row in actual]==[10000,10001]


@pytest.mark.parametrize('saved_count',[1,4,5,6])
def test_resume_replays_original_variable_shape_batches(tmp_path,saved_count):
    source=write_rows(tmp_path/'source.jsonl',source_rows())
    output=tmp_path/'run'
    args=arguments(output,source)
    evaluator.main(args)
    expected=records(output)
    saved=expected[:saved_count]
    write_rows(output/'samples.jsonl',saved)
    evaluator.main(args+['--resume'])
    actual=records(output)
    keys=('sample_id','draw_id','token_ids','prefix_length','generated_length',
          'batch_id','batch_size','document_id','chunk_id','input_index','continuation_file_sha256')
    assert [[row[key] for key in keys] for row in actual]==[[row[key] for key in keys] for row in expected]
    assert actual[:saved_count]==saved


@pytest.mark.parametrize('field',['document_id','prefix','generated_token'])
def test_resume_rejects_identity_prefix_or_replayed_token_changes_before_append(tmp_path,field):
    source=write_rows(tmp_path/'source.jsonl',source_rows())
    output=tmp_path/'run'
    args=arguments(output,source)
    evaluator.main(args)
    saved=records(output)[:1]
    if field=='document_id':
        saved[0]['document_id']='wrong-article'
    else:
        index=0 if field=='prefix' else -1
        saved[0]['token_ids'][index]=(saved[0]['token_ids'][index]+1)%16
    path=write_rows(output/'samples.jsonl',saved)
    before=path.read_bytes()
    with pytest.raises(ValueError,match='Stored continuation|Replayed partial batch'):
        evaluator.main(args+['--resume'])
    assert path.read_bytes()==before


def test_changed_input_file_rejected_by_resume_manifest(tmp_path):
    inputs=source_rows()
    source=write_rows(tmp_path/'source.jsonl',inputs)
    output=tmp_path/'run'
    args=arguments(output,source)
    evaluator.main(args)
    inputs[0]['reference_continuation_ids'][0]=9
    write_rows(source,inputs)
    before=(output/'samples.jsonl').read_bytes()
    with pytest.raises(ValueError,match='Resume manifest'):
        evaluator.main(args+['--resume'])
    assert (output/'samples.jsonl').read_bytes()==before


def test_score_gpt2_conditions_on_full_prefix_but_only_scores_suffix(monkeypatch):
    seen=[]

    class Scorer:
        config=SimpleNamespace(n_positions=16,_commit_hash='fixture')
        def to(self,device):
            return self
        def eval(self):
            return self
        def __call__(self,ids):
            seen.append(ids.tolist())
            # Distinct logits per position make an incorrect target slice visible.
            logits=torch.arange(ids.shape[1]*5,dtype=torch.float32).reshape(1,ids.shape[1],5)/10
            return SimpleNamespace(logits=logits)

    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args,**kwargs:Scorer())))
    row={'sample_id':0,'token_ids':[4,3,2,1,0],'prefix_length':3}
    result=evaluator.score_gpt2([row],'cpu')
    assert seen==[[row['token_ids']]]
    logits=torch.arange(25,dtype=torch.float32).reshape(5,5)/10
    expected=torch.nn.functional.cross_entropy(logits[2:4],torch.tensor([1,0]),reduction='sum')
    assert result['scored_tokens']==2
    assert result['samples'][0]['scored_tokens']==2
    assert result['total_nll']==pytest.approx(float(expected))
