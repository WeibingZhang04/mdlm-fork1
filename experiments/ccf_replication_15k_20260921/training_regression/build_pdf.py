from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
root=Path(__file__).resolve().parents[3]
out=root/'output/pdf/ccf-training-regression-audit.pdf';out.parent.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleAudit',fontName='Helvetica-Bold',fontSize=21,leading=25,textColor=colors.HexColor('#17364B'),spaceAfter=8))
styles.add(ParagraphStyle(name='BodyAudit',fontName='Helvetica',fontSize=10.1,leading=14.5,spaceAfter=9))
styles.add(ParagraphStyle(name='SmallAudit',fontName='Helvetica',fontSize=8.5,leading=11.5,textColor=colors.HexColor('#526574'),spaceAfter=8))
styles.add(ParagraphStyle(name='LabelAudit',fontName='Helvetica-Bold',fontSize=11,leading=15,spaceBefore=9,spaceAfter=6,textColor=colors.HexColor('#17364B')))
story=[]
def p(s,style='BodyAudit'):story.append(Paragraph(s,styles[style]))
p('Why did the fresh CCF runs change?','TitleAudit')
p('Training audit  |  21 September 2026  |  Confirmed GPU replay','SmallAudit')
p('<b>The added monitoring callback changed the numerical training setup.</b> The fresh runs were not an equivalent replication of the earlier training. The runtime optimizations are not implicated by the controlled replay.')
p('What happened','LabelAudit')
p('The old run first computed and cached its positional rotations during BF16 training. The new startup diagnostic computed that cache in FP32 before training began. The model reused whichever cache was created first, even though the backbone weights and saved settings were identical.')
p('Generation initializes this cache in FP32 (the transformer blocks still use BF16 internally). Historical adapters therefore trained and generated with different cache precision. Fresh monitored runs use FP32 for both. This is a numerical execution change, not an architecture change.')
p('Controlled replay: identical weights, data and seed','LabelAudit')
rows=[['Setup','Cache','Backbone NLL*'],['Old code; no startup probe','BF16','5.53275'],['New code; no startup probe','BF16','5.53275'],['New code; current startup probe','FP32','3.88481'],['New code; probe run under BF16','BF16','5.53275']]
t=Table(rows,colWidths=[294,60,130],hAlign='LEFT')
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17364B')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,1),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9.5),('LEADING',(0,0),(-1,-1),12),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F5F8'),colors.white]),('ALIGN',(2,0),(2,-1),'RIGHT'),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
story.append(t);story.append(Spacer(1,8))
p('*Frozen-backbone negative log-likelihood on training batch 10; <b>not generation PPL</b>. The old and current-probe values exactly reproduce the recorded training losses for the frozen backbone. No optimizer updates in this replay. All initial head hashes and ten input batches match; all ten loss/metric records match exactly across the BF16-cache cases. Job 1557313 completed on RTX 6000 Ada.','SmallAudit')
p('What this means for the results','LabelAudit')
p('Data provenance, backbone, tokenizer, batch size, learning rate and objective settings match. Earlier Basic runs were resumed in stages; old Separate R8 FD was also fresh, so resume history cannot explain this discrepancy. The earlier audit missed the callback plus mixed-precision startup.')
p('<b>The measured PPL losses to MDLM remain real.</b> This finding establishes a training confound; it does not yet quantify how much it caused the PPL reversal. Preserve both sets of results and disclose the difference. The next control should fix cache precision explicitly and verify that enabling monitoring cannot change training.')
p('No architecture or model-source changes were made in this audit. Existing jobs and historical results were preserved. Full evidence and replay script accompany the Markdown audit.','SmallAudit')
def footer(c,d):
 c.setStrokeColor(colors.HexColor('#D7E1E8'));c.line(54,43,558,43);c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#526574'));c.drawString(54,30,'CCF / g-experiments - numerical training audit');c.drawRightString(558,30,str(d.page))
SimpleDocTemplate(str(out),pagesize=(612,792),rightMargin=54,leftMargin=54,topMargin=42,bottomMargin=53).build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
