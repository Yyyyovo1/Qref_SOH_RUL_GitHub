"""Verify the packaged bytes using the release manifest; no third-party dependencies."""
from pathlib import Path
import csv, hashlib
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    with (ROOT/'provenance/SHA256SUMS.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    failures=[]
    for row in rows:
        p=ROOT/row['path']
        if not p.is_file() or digest(p)!=row['sha256']:failures.append(row['path'])
    if failures:raise SystemExit('FAILED: '+str(failures))
    print('PASS:',len(rows),'packaged files match SHA-256')
if __name__=='__main__':main()
