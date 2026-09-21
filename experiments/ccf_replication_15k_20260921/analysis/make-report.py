from pathlib import Path
import json,gzip,datetime
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
w=Path(__file__).parent;out=Path('outputs');out.mkdir(exist_ok=True)
s=json.loads(gzip.decompress((w/'snapshot.json.gz').read_bytes()));a=json.loads((w/'topology-analysis.json').read_text());state=s['state']
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyCCF',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8,textColor=colors.HexColor('#243247')));styles.add(ParagraphStyle(name='SmallCCF',parent=styles['BodyCCF'],fontSize=8.3,leading=11));styles['Title'].fontName='Helvetica-Bold';styles['Title'].fontSize=23;styles['Title'].leading=27;styles['Title'].textColor=colors.HexColor('#132B47');styles['Heading2'].fontSize=13;styles['Heading2'].leading=17;styles['Heading2'].textColor=colors.HexColor('#132B47')
story=[]
def p(t,style='BodyCCF'):story.append(Paragraph(t,styles[style]))
def h(t):p(t,'Heading2')
def table(rows,widths):
 cells=[[Paragraph(str(x),styles['SmallCCF']) for x in row] for row in rows]
 t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EDF5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#A5B5C8')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#DAE1E8')),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.append(t);story.append(Spacer(1,9))
when=s['collected_utc'][:16].replace('T',' ')+' UTC'
p('CCF: audit and experiment launch','Title')
p('Progress report | '+when+' | Final 15k results are still pending.','SmallCCF')
h('g-experiments passed the replication audit')
p('Use the isolated copy of <b>g-experiments, commit 2051502</b>. The pinned training data, tokenizer, released frozen MDLM backbone, validation loader, generation harness and PPL scorer match the previous experiment code. All 61 targeted tests passed.')
p('On real data, all eight arms had identical losses; maximum gradient difference was 3.73e-9. Old/new code generated identical tokens and actual model-call counts at 8/16/32 denoising steps for every arm. A larger replay also matched all <b>120 generated sequences and their PPL scores</b>. This validates the tested paths, not every possible input.')
p('No evidence of ground-truth leakage in generation: candidates and topology use corrupted context and time. Clean tokens supply training targets and the topology teacher only. This audit does not certify that public validation text was absent from backbone pretraining.')
h('Same-GPU replay: old code = new code')
replay=next(g for g in s['generation'] if g['path']=='legacy_replay/new/generation')
trows=[['20 samples per cell; lower PPL is better','8 steps','16 steps','32 steps']]
for mode,label in [('factorized','MDLM baseline'),('structured_joint','Basic FD, existing 6k checkpoint')]:
 vals={g['requested_nfe_budget']-1:g['reference_lm']['perplexity'] for g in replay['summary']['groups'] if g['sampling_mode']==mode}
 trows.append([label,*[f'{vals[n]:.2f}' for n in [8,16,32]]])
table(trows,[290,70,70,70])
p('Replay GPU: RTX 6000 Ada. These values differ from historical H200 results even with old code. Every new arm evaluation therefore includes its own same-GPU MDLM baseline. The replay uses one 6k checkpoint, not the best checkpoint selected from a sweep.','SmallCCF')
h('Replication protocol')
table([['Arms','Training','Generation evaluation'],['Basic FF, FD, DF, DD; separate R8 FD/DD; separate R16 FD/DD','8 fresh runs to 15,000 updates','4, 8, 16, 32 denoising steps'],['Checkpoint curve','Export every 1,000 updates','20 samples at 1-7k, 10k, 12k, 15k'],['Larger checks','Preselected 7k, 10k, 15k','100 samples per checkpoint and denoising budget']],[210,120,170])
p('<b>Labels:</b> first letter = topology, second = factors; F = fixed form, D = context-dependent. FF replaces the old SS label; fixed factors still have learned parameters. R8/R16 are factor ranks. A training update processes four sequences. A denoising step is a reverse-time update. Nominal NFE budget is steps + 1; actual model calls are recorded separately.','SmallCCF')
story.append(PageBreak())
p('What the first diagnostics show','Title')
p('Completed: existing 6k FD/DD checkpoints, three factor families, 16 fixed validation chunks, lengths 128/512/1024, four mask rates, six graph interventions. These are conditional likelihood tests, not generation PPL.','SmallCCF')
h('More edges alone is unlikely to solve DD')
lookup={(r['variant'],r['length'],r['intervention']):r for r in a['summary']}
rows=[['At length 1,024','FD edge density','DD edge density','DD with fixed graph: NLL change']]
for prefix,label in [('basic','Basic'),('separate_r8','Separate R8'),('separate_r16','Separate R16')]:
 fd=lookup[(prefix+'_fixed_dynamic',1024,'native')];dd=lookup[(prefix+'_dynamic_dynamic',1024,'native')];fix=lookup[(prefix+'_dynamic_dynamic',1024,'fixed_graph')]
 rows.append([label,f"{fd['edge_fraction_of_tree']:.1%}",f"{dd['edge_fraction_of_tree']:.1%}",f"{fix['nll_delta_vs_native']:+.4f} nats/token"])
table(rows,[100,115,115,170])
p('Density = selected edges / (masked tokens - 1). Negative NLL change is better. DD is sparser, yet raising its component cap from 32 to 128 makes NLL worse in all three families. Using the fixed graph with the same DD weights improves NLL. Edge placement and component structure deserve attention before adding edges.')
p('DD edge ranking beats a uniform predictor, but global anchor selection is near uniform and slot routing is slightly worse than uniform on this panel. That suggests a weak part of the topology mechanism. It does not establish that low loss weight is the cause: the teacher target also varies with the randomly revealed source token.')
h('Why a flat loss can be misleading')
p('The optimized loss is <b>joint masked-token NLL + 0.1 x topology cross-entropy</b> for DF/DD; FF/FD use joint NLL only. The factorized auxiliary coefficient is zero. Joint NLL includes both unary and pairwise terms, and much of its level comes from the frozen backbone. Topology cross-entropy includes teacher entropy, so it need not approach zero.')
p('New logs separate joint NLL, topology loss, gain over the frozen backbone, gradient norms and edge counts. A fixed-mask validation panel tracks joint likelihood versus the product of the same exact marginals. PPL curves will determine whether longer training helps generation or plateaus; that answer is not available yet.')
h('Follow-up experiments and paper improvements')
p('Queued: topology weights <b>0.03 / 0.10 / 0.30</b>, each continuing the same Basic DD 6k checkpoint for 2,000 updates. Each has 100-sample generation checks at 8 and 32 denoising steps, including joint versus exact-marginal sampling. The main 15k runs separately test longer training.')
p('The paper motivates separate endpoint factors, joint-versus-marginal controls, and evaluation across mask rates. Next useful controls are a parameter-matched independent adapter and marginal-preserving dependence shrinkage. Keep repetition-1/2/4, distinct-2/4, output length and EOS position beside PPL, since PPL alone does not measure diversity or coherence.')
p('Scope: one training seed; exploratory diagnostics on a small fixed WikiText panel. Fresh continuous 15k runs are not identical trajectories to old runs that restarted at intermediate checkpoints. Preserve this distinction when interpreting changes.','SmallCCF')
p('Operations: generic compatible GPUs, no H200/node pin. Original source and old runs remain unchanged; no other person\'s jobs or files were modified. Audit launcher retries are preserved. Training array '+state['jobs']['training']+'; evaluation '+state['jobs']['evaluation']+'; weight sweep '+state['jobs']['weights']+'.','SmallCCF')
def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#CCD6E1'));c.line(48,40,548,40);c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#64748B'));c.drawString(48,28,'CCF | Interim audit and launch report | 21 September 2026');c.drawRightString(548,28,str(doc.page))
doc=SimpleDocTemplate(str(out/'ccf-g-experiments-audit-and-launch.pdf'),pagesize=(596,842),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=52,title='CCF g-experiments audit and launch',author='Nina - experiment audit')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(out/'ccf-g-experiments-audit-and-launch.pdf')
