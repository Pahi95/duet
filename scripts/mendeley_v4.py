"""Restore Mendeley Cite controls and the matching Office add-in citation cache.

The bibliographic items come from the author's original document. Library IDs
are preserved; the added lme4 reference is a temporary item with full metadata.
This performs package-level reconstruction, not an interactive add-in test.
"""
import base64
import copy
import json
import re
import uuid
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
REL='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
NS={'w':W}
PATTERN=re.compile(r'\[\d+(?:[–-]\d+)?\](?:,\s*\[\d+(?:[–-]\d+)?\])*')

def xml(e):
    return E.tostring(e,encoding='UTF-8',xml_declaration=True,standalone=True)

def restore(path,source):
    items=json.loads((source/'mendeley_items.json').read_text(encoding='utf8'))
    with ZipFile(path) as z:parts={n:z.read(n) for n in z.namelist()}
    root=E.fromstring(parts['word/document.xml']);cache=[];used=set()
    for p in root.findall('w:body/w:p',NS):
        runs=p.findall('w:r',NS)
        spans=[];offset=0
        for r in runs:
            text=''.join(r.xpath('./w:t/text()',namespaces=NS))
            spans.append((offset,offset+len(text),r,text));offset+=len(text)
        text=''.join(s[3] for s in spans)
        matches=list(PATTERN.finditer(text))
        if not matches:continue
        assert all(e.tag in [f'{{{W}}}pPr',f'{{{W}}}r'] for e in p)
        def fragments(lo,hi):
            out=[]
            for start,end,old,txt in spans:
                a,b=max(lo,start),min(hi,end)
                if b<=a:continue
                r=E.Element(f'{{{W}}}r')
                props=old.find('w:rPr',NS)
                if props is not None:r.append(copy.deepcopy(props))
                t=E.SubElement(r,f'{{{W}}}t');t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=txt[a-start:b-start]
                out.append(r)
            return out
        for r in runs:p.remove(r)
        last=0
        for m in matches:
            for r in fragments(last,m.start()):p.append(r)
            numbers=[]
            for first,end in re.findall(r'(\d+)(?:[–-](\d+))?',m[0]):
                numbers.extend(range(int(first),int(end or first)+1))
            assert all(str(n) in items for n in numbers),m[0]
            used.update(numbers)
            identity=str(uuid.uuid5(uuid.NAMESPACE_URL,f'DUET-V4-citation-{len(cache)}-{m[0]}'))
            data={'citationID':'MENDELEY_CITATION_'+identity,'properties':{'noteIndex':0},'isEdited':False,'manualOverride':{'isManuallyOverridden':False,'citeprocText':m[0],'manualOverrideText':''},'citationItems':[items[str(n)] for n in numbers]}
            tag='MENDELEY_CITATION_v3_'+base64.b64encode(json.dumps(data,ensure_ascii=False,separators=(',',':')).encode()).decode()
            s=E.SubElement(p,f'{{{W}}}sdt');pr=E.SubElement(s,f'{{{W}}}sdtPr')
            E.SubElement(pr,f'{{{W}}}tag').set(f'{{{W}}}val',tag)
            E.SubElement(pr,f'{{{W}}}id').set(f'{{{W}}}val',str(100000+len(cache)))
            content=E.SubElement(s,f'{{{W}}}sdtContent')
            for r in fragments(m.start(),m.end()):content.append(r)
            data['citationTag']=tag;cache.append(data);last=m.end()
        for r in fragments(last,len(text)):p.append(r)
    assert used==set(range(1,33)),used
    bib=root.xpath('//w:sdt[w:sdtPr/w:tag/@w:val="MENDELEY_BIBLIOGRAPHY"]',namespaces=NS)
    assert len(bib)==1 and len(bib[0].findall('w:sdtContent/w:p',NS))==32
    parts['word/document.xml']=xml(root)
    web=E.fromstring((source/'mendeley_webextension.xml').read_bytes())
    for prop in web.findall('.//{*}property'):
        if prop.get('name')=='MENDELEY_CITATIONS':prop.set('value',json.dumps(cache,ensure_ascii=False,separators=(',',':')))
    parts['word/webextensions/webextension1.xml']=xml(web)
    parts['word/webextensions/taskpanes.xml']=b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><wetp:taskpanes xmlns:wetp="http://schemas.microsoft.com/office/webextensions/taskpanes/2010/11"><wetp:taskpane dockstate="right" visibility="0" width="350" row="3"><wetp:webextensionref xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="rId1"/></wetp:taskpane></wetp:taskpanes>'''
    rel=E.Element(f'{{{REL}}}Relationships',nsmap={None:REL})
    E.SubElement(rel,f'{{{REL}}}Relationship',Id='rId1',Type='http://schemas.microsoft.com/office/2011/relationships/webextension',Target='webextension1.xml')
    parts['word/webextensions/_rels/taskpanes.xml.rels']=xml(rel)
    rel=E.fromstring(parts['_rels/.rels'])
    E.SubElement(rel,f'{{{REL}}}Relationship',Id='rIdMendeley',Type='http://schemas.microsoft.com/office/2011/relationships/webextensiontaskpanes',Target='word/webextensions/taskpanes.xml')
    parts['_rels/.rels']=xml(rel)
    ct=E.fromstring(parts['[Content_Types].xml'])
    for name,typ in [('webextension1.xml','webextension'),('taskpanes.xml','webextensiontaskpanes')]:
        E.SubElement(ct,f'{{{CT}}}Override',PartName='/word/webextensions/'+name,ContentType='application/vnd.ms-office.'+typ+'+xml')
    parts['[Content_Types].xml']=xml(ct)
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        for name,content in parts.items():z.writestr(name,content)
    print(f'{path.name}: {len(cache)} Mendeley citation controls, 32 references and synchronized add-in cache.')
