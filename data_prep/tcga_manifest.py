"""GDC open-access TCGA diagnostic slides (FFPE, DX) for a cancer group -> manifest.tsv (file_id, file_name, size, md5, project, case).
usage: python3 tcga_manifest.py OUTDIR PROJECT [PROJECT ...]   e.g. ... data/tcga_rcc TCGA-KIRC TCGA-KIRP TCGA-KICH  (stdlib only)"""
import sys, os, json, urllib.request
out, projects = sys.argv[1], sys.argv[2:]; os.makedirs(f'{out}/raw', exist_ok=True)
flt = {"op": "and", "content": [
    {"op": "in", "content": {"field": "cases.project.project_id", "value": projects}},
    {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
    {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
    {"op": "=", "content": {"field": "access", "value": "open"}}]}
body = json.dumps({"filters": flt, "fields": "file_id,file_name,file_size,md5sum,cases.project.project_id,cases.submitter_id",
                   "format": "JSON", "size": 10000}).encode()
req = urllib.request.Request("https://api.gdc.cancer.gov/files", body, {"Content-Type": "application/json"})
hits = json.load(urllib.request.urlopen(req, timeout=120))["data"]["hits"]
rows = sorted((h["file_id"], h["file_name"], h["file_size"], h["md5sum"], h["cases"][0]["project"]["project_id"], h["cases"][0]["submitter_id"]) for h in hits)
with open(f'{out}/manifest.tsv', 'w') as f:
    f.write('file_id\tfile_name\tsize\tmd5\tproject\tcase\n'); [f.write('\t'.join(map(str, r)) + '\n') for r in rows]
from collections import Counter
print(len(rows), 'slides', f'{sum(r[2] for r in rows) / 1e12:.2f} TB', dict(Counter(r[4] for r in rows)), 'cases', len({r[5] for r in rows}))
