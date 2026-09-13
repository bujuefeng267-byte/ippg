"""Run the installed test suite and save a content-bound release receipt."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from datetime import datetime,timezone

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();folder=Path(__file__).resolve().parent
    if a.out.exists():raise FileExistsError(a.out)
    files=[x for x in sorted(folder.glob('*.py')) if x.name!='check_release_v27.py']
    hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(folder),'-p','test_*.py','-v'],
        capture_output=True,text=True)
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_directory=str(folder),
        interpreter=sys.executable,exit_code=result.returncode,source_hashes=hashes,
        stdout=result.stdout,stderr=result.stderr,
        scope='Synthetic regression and contract tests. Does not establish real-video accuracy.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(result.stdout);print(result.stderr);print('Receipt:',a.out)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
