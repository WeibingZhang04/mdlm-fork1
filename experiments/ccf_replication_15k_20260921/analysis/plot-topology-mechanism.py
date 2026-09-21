from pathlib import Path
import gzip,json,statistics,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
w=Path(__file__).parent;s=json.loads(gzip.decompress((w/'snapshot.json.gz').read_bytes()));rows=s['files']['diagnostics/topology6k/records.jsonl']
ix={(r['variant'],r['sequence_length'],r['mask_rate'],r['sample'],r['intervention']):r for r in rows}
rates=[.25,.5,.75,.9];palette=['#1763A6','#A85518','#597C2E'];labels=['Basic','Separate R8','Separate R16'];prefixes=['basic','separate_r8','separate_r16'];data=[]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
for pref,label,color in zip(prefixes,labels,palette):
 vals=[];deltas=[]
 for rate in rates:
  group=[r for r in rows if r['variant']==pref+'_dynamic_dynamic' and r['sequence_length']==1024 and r['mask_rate']==rate and r['intervention']=='native']
  iso=100*statistics.mean(r['isolated_fraction'] for r in group)
  delta=statistics.mean(ix[(r['variant'],1024,rate,r['sample'],'fixed_graph')]['joint_nll']-r['joint_nll'] for r in group)
  vals.append(iso);deltas.append(delta);data.append({'model':label,'mask_rate':rate,'isolated_masked_tokens_pct':iso,'fixed_graph_minus_native_nll':delta,'samples':16})
 axes[0].plot([r*100 for r in rates],vals,'o-',color=color,label=label+' DD',lw=1.8)
 axes[1].plot([r*100 for r in rates],deltas,'o-',color=color,label=label,lw=1.8)
fd=[100*statistics.mean(r['isolated_fraction'] for r in rows if r['variant']=='basic_fixed_dynamic' and r['sequence_length']==1024 and r['mask_rate']==rate and r['intervention']=='native') for rate in rates]
axes[0].plot([r*100 for r in rates],fd,'--',color='#737D8C',label='Fixed graph (all families)')
axes[0].set(title='DD loses connections at low masking',ylabel='Isolated masked tokens (%)',ylim=(-.5,26));axes[0].legend(fontsize=8,frameon=False)
axes[1].axhline(0,color='#737D8C',lw=.7);axes[1].set(title='Fixed graph improves DD conditional NLL',ylabel='Fixed graph − learned graph NLL (nats/token)');axes[1].text(.98,.08,'Below zero = fixed graph improves NLL',transform=axes[1].transAxes,ha='right',fontsize=8,color='#465469')
for ax in axes:ax.set_xlabel('Tokens masked (%)');ax.set_xticks([25,50,75,90]);ax.grid(alpha=.15)
fig.suptitle('Existing 6k checkpoints • length 1,024 • 16 fixed validation chunks',fontsize=11)
fig.savefig('outputs/ccf-dynamic-topology-diagnosis.png',dpi=180)
fig.savefig('outputs/ccf-dynamic-topology-diagnosis-figure.pdf')
(w/'topology-mechanism-summary.json').write_text(json.dumps(data,indent=2))
