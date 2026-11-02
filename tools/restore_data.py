"""Restore losslessly compressed data. Python standard library only."""
from pathlib import Path
import gzip, hashlib, json, shutil, tempfile
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    records=json.loads((ROOT/'provenance/PACKED_DATA.json').read_text('utf8'))
    for record in records:
        target=ROOT/record['path']
        if target.exists():
            if digest(target)!=record['sha256']:raise SystemExit('REFUSING TO OVERWRITE: '+record['path'])
            print('OK existing',record['path']);continue
        with tempfile.TemporaryFile() as combined:
            for part in record['parts']:
                p=ROOT/part['path']
                if digest(p)!=part['sha256']:raise SystemExit('BAD PART: '+part['path'])
                with p.open('rb') as f:shutil.copyfileobj(f,combined)
            combined.seek(0)
            # A temporary file in the destination directory permits atomic commit.
            with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as output:
                temporary=Path(output.name)
                try:
                    with gzip.GzipFile(fileobj=combined,mode='rb') as data:shutil.copyfileobj(data,output)
                except BaseException:
                    output.close();temporary.unlink();raise
            try:
                if temporary.stat().st_size!=record['bytes'] or digest(temporary)!=record['sha256']:
                    raise RuntimeError('RESTORED HASH MISMATCH: '+record['path'])
                if target.exists():raise RuntimeError('Target appeared during restore: '+record['path'])
                temporary.rename(target)
            finally:
                temporary.unlink(missing_ok=True)
        print('RESTORED',record['path'])
    print('Data restore complete:',len(records),'files')
if __name__=='__main__':main()
