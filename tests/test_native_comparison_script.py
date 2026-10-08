"""Dry-run the real comparison shell script with inert model/scorer commands."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('cap,native_cap', [('500','500'),('none','null')])
def test_five_method_script_shares_settings_and_isolates_native_cache(tmp_path,cap,native_cap):
    conda = tmp_path/'conda.sh'
    conda.write_text('conda() { return 0; }\n')
    base = tmp_path/'data'
    (base/'runs').mkdir(parents=True)
    bindir = tmp_path/'bin'
    bindir.mkdir()
    mock = bindir/'python'
    mock.write_text('#!'+sys.executable+'\n'+r'''
import json,os,sys
from pathlib import Path
args=[arg for arg in sys.argv[1:] if arg != '-u']
if args[0]=='-':
    os.execv(sys.executable,[sys.executable]+args)
with open(os.environ['COMMAND_LOG'],'a') as stream:
    stream.write(json.dumps({'args':args,'hf_home':os.environ.get('HF_HOME')})+'\n')
if args[0]=='main.py':
    config=dict(arg.split('=',1) for arg in args[1:])
    out=Path(config['eval.sample_output_dir'])
    (out/'samples.jsonl').write_text('{}\n')
    (out/'metrics.json').write_text(json.dumps({'seconds_per_sample':.2,'post_load_seconds_per_sample':.35,'backbone_calls_per_batch':33}))
else:
    out=Path(args[args.index('--output')+1])
    out.mkdir(exist_ok=True)
    if '--score-only' in args:
        assert (out/'samples.jsonl').exists()
    else:
        (out/'metrics.json').write_text(json.dumps({'seconds_per_sample':.4,'post_load_seconds_per_sample':.65,'backbone_calls_per_sample':33}))
    (out/'gpt2-large.json').write_text(json.dumps({'perplexity':100.}))
''')
    mock.chmod(0o755)
    source=(ROOT/'quick_ppl_check_capped_vocab.sh').read_text()
    source=source.replace('cd /u401/n23zhang/crf-rework/mdlm-fork1',f'cd "{tmp_path}"')
    source=source.replace('source /opt/anaconda3/etc/profile.d/conda.sh',f'source "{conda}"')
    source=source.replace('base=/u401/n23zhang/rework-data',f'base="{base}"')
    script=tmp_path/'comparison.sh'
    script.write_text(source)
    log=tmp_path/'commands.jsonl'
    env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
         'VOCAB_CAP':cap,'COMMAND_LOG':str(log)}
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    commands=[json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands)==6  # Native generation, native score-only, four custom methods.
    native=dict(arg.split('=',1) for arg in commands[0]['args'][1:])
    assert native['sampling.vocab_cap']==native_cap
    assert native['sampling.steps']=='32' and native['sampling.num_sample_batches']=='60'
    assert native['sampling.warmup_batches']=='1' and native['loader.eval_batch_size']=='1'
    assert native['sampling.noise_removal']=='True'
    assert native['sampling.predictor']=='ddpm_cache'
    assert commands[0]['hf_home']==str(base/'original_reproduction/cache/mdlm/huggingface')
    assert '--score-only' in commands[1]['args']
    assert all(command['hf_home']==str(base/'hf-cache') for command in commands[1:])
    for command in commands[2:]:
        args=command['args']
        assert args[args.index('--vocab-cap')+1]==cap
        assert args[args.index('--samples')+1]=='60'
        assert args[args.index('--steps')+1]=='32'
        assert args[args.index('--sampler')+1]=='ddpm_cache'
        assert '--noise-removal' in args and '--no-noise-removal' not in args
        assert args[args.index('--sampling-eps')+1]=='1e-5'
        assert args[args.index('--temperature')+1]=='1'
        assert '--stage-timing' not in args
        assert args[args.index('--warmup')+1]=='1'
    assert 'Total for all five evaluations:' in result.stdout
    assert 'native_mdlm' in result.stdout and '33.0' in result.stdout
    assert 'vanilla' in result.stdout and '33.0' in result.stdout
    assert 'All five methods use DDPM-cache' in result.stdout
    assert 'Post-load s/sample' in result.stdout
    assert '0.3500' in result.stdout and '0.6500' in result.stdout
