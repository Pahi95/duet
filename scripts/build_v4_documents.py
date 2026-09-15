"""Build V4 Word files only from publication/v4 (no original expression data needed).

python scripts/build_v4_documents.py --output ../manuscript
Optional --redline-against PATH creates a third copy marking new words in red.
Requires python-docx and pandas. The original documents are never overwritten.
"""
import argparse, difflib, json, re
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_COLOR_INDEX


def fresh(landscape=False):
    d=Document()
    s=d.sections[0]
    s.page_width=Inches(11.69 if landscape else 8.27)
    s.page_height=Inches(8.27 if landscape else 11.69)
    s.top_margin=s.bottom_margin=Inches(.75)
    s.left_margin=s.right_margin=Inches(.65)
    for name in ['Normal','Heading 1','Heading 2','Title']:
        st=d.styles[name]
        st.font.name='Times New Roman'
        st.font.color.rgb=RGBColor(0,0,0)
        st.font.size=Pt(11 if name=='Normal' else 13)
        st.paragraph_format.line_spacing=2 if not landscape else 1.15
        st.paragraph_format.space_after=Pt(5)
    ln=OxmlElement('w:lnNumType'); ln.set(qn('w:countBy'),'1'); ln.set(qn('w:restart'),'continuous')
    if not landscape: s._sectPr.append(ln)
    p=s.footer.paragraphs[0]; p.alignment=1
    fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); p._p.append(fld)
    d.core_properties.author=''
    d.core_properties.last_modified_by=''
    d.core_properties.title='DUET manuscript V4' if not landscape else 'DUET supplementary tables V4'
    return d


def inline(p,text):
    for bit in re.split(r'(\*\*.*?\*\*|==.*?==|`.*?`)',text):
        r=p.add_run(bit[2:-2] if bit.startswith(('**','==')) else bit.strip('`'))
        if bit.startswith('**'):r.bold=True
        if bit.startswith('=='):r.font.highlight_color=WD_COLOR_INDEX.YELLOW
        if bit.startswith('`'):r.font.name='Consolas'


def table(d,caption,df,landscape=False):
    # Split wide tables into readable column panels, repeating row identifiers.
    maxcols=7 if landscape else 5
    if landscape and len(df.columns)>maxcols:
        df=df.copy()
        df.insert(0,'Row ID',range(1,len(df)+1))
    cols=list(df.columns)
    groups=[cols] if len(cols)<=maxcols else [cols[:2]+cols[i:i+maxcols-2] for i in range(2,len(cols),maxcols-2)]
    for part,group in enumerate(groups):
        p=d.add_paragraph(); inline(p,caption+(f' (column panel {part+1}/{len(groups)})' if len(groups)>1 else ''))
        p.paragraph_format.keep_with_next=True
        p.paragraph_format.line_spacing=1.15
        t=d.add_table(rows=1,cols=len(group)); t.style='Table Grid'
        hdr=t.rows[0]._tr.get_or_add_trPr(); hdr.append(OxmlElement('w:tblHeader'))
        for cell,c in zip(t.rows[0].cells,group):cell.text=str(c).replace('_',' ')
        for values in df[group].itertuples(index=False,name=None):
            row=t.add_row(); row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for cell,value in zip(row.cells,values):
                cell.text='' if pd.isna(value) else (f'{value:.4g}' if isinstance(value,float) else str(value))
        for ri,row in enumerate(t.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing=1
                    p.paragraph_format.space_after=Pt(3)
                    p.paragraph_format.space_before=Pt(3)
                    for r in p.runs:r.font.size=Pt(9 if landscape else 10);r.bold=ri==0
        d.add_paragraph().paragraph_format.space_after=Pt(3)


def vancouver(s):
    m=re.fullmatch(r'\[(\d+)\](.*?)“(.*?),” (.*?), vol\. (.*?), (?:no\. (.*?), )?pp?\. (.*?), (\d{4}), doi: (.*?)\.',s)
    if not m:return s
    n,authors,title,journal,vol,issue,pages,year,doi=m.groups()
    aa=[]
    for a in authors.strip(' ,').replace(', and ', ', ').replace(' and ', ', ').split(', '):
        a=a.removeprefix('and ')
        suffix=' et al.' if ' et al.' in a else ''
        a=a.replace(' et al.','')
        am=re.match(r'^((?:(?:[A-Z][.\- ]+)|(?:[A-Z] ))+)(.+)$',a)
        aa.append((am[2]+' '+re.sub(r'[. ]','',am[1]) if am else a)+suffix)
    return f'[{n}] '+', '.join(aa)+f'. {title}. {journal} {year};{vol}'+(f'({issue})' if issue else '')+f':{pages}. doi:{doi}.'


def build(source,output,redline=None):
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((source/'tables.json').read_text(encoding='utf-8'))
    tabs={t['id']:t for t in meta}
    doc=fresh()
    for line in (source/'manuscript.md').read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        if line.startswith('@@TABLE'):
            t=tabs[line.split()[1]];table(doc,t['caption'],pd.read_csv(source/t['file']));continue
        if line.startswith('@@FIGURE'):
            p=doc.add_paragraph();p.paragraph_format.line_spacing=1
            p.add_run().add_picture(str(source/'figures'/line.split()[1]),width=Inches(6.9))
            # Figures may be followed by long captions: let captions flow instead of forcing blank pages.
            continue
        if line.startswith('@@REFERENCES'):
            for ref in (source/'references.txt').read_text(encoding='utf-8').splitlines():doc.add_paragraph(vancouver(ref))
            continue
        style='Normal'
        for prefix,st in [('### ','Heading 2'),('## ','Heading 1'),('# ','Title')]:
            if line.startswith(prefix):line=line[len(prefix):];style=st;break
        p=doc.add_paragraph(style=style);inline(p,line.removeprefix('- '))
    main=output/'DUET_manuscript_revised_MO_v4.docx';doc.save(main)
    sup=fresh(True)
    sup.add_paragraph('Additional file 2. Supplementary tables — V4',style='Title')
    sup.add_paragraph('Machine-readable CSVs accompany every table. Wide tables are split into column panels with repeated identifiers. Memory fields ending in _mb in archived source files are measured in MiB (bytes / 2²⁰).')
    for t in meta:
        if t['supplement']:table(sup,t['caption'],pd.read_csv(source/t['file']),True)
    sup.save(output/'DUET_additional_file_2_tables_v4.docx')
    if redline:
        old=Document(redline)
        oldpars=[p.text for p in old.paragraphs if p.text.strip()]
        for p in doc.paragraphs:
            if not p.text.strip() or not p.runs:continue
            candidates=sorted(oldpars,key=lambda x:difflib.SequenceMatcher(None,x,p.text).quick_ratio(),reverse=True)[:3]
            best=max(candidates,key=lambda x:difflib.SequenceMatcher(None,x.split(),p.text.split()).ratio())
            new=p.text
            if new==best:continue
            # Word-token diff preserves equal text and marks only additions/replacements red.
            a=re.findall(r'\S+\s*',best);b=re.findall(r'\S+\s*',new)
            p.clear()
            for tag,_,_,j1,j2 in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
                if tag=='delete':continue
                r=p.add_run(''.join(b[j1:j2]))
                if tag!='equal':r.font.color.rgb=RGBColor(192,0,0)
        doc.save(output/'DUET_manuscript_revised_MO_v4_red.docx')
    print('Wrote V4 manuscript, supplementary tables'+(' and additions-marked copy.' if redline else '.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'publication/v4')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--redline-against',type=Path)
    a=ap.parse_args();build(a.source,a.output,a.redline_against)
