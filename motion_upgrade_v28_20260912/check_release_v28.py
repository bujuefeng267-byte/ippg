"""Run the installed synthetic/regression suite and save exact source identity."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    folder=Path(__file__).resolve().parent
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(folder.glob('*.py'))}
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(folder),'-p','test_*.py','-v'],capture_output=True,text=True)
    for path,h in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    a.out.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,
        source_hashes=hashes,scope='Implementation and input/output contracts; real performance is evaluated separately.'),ensure_ascii=False,indent=2))
    print(result.stdout);print(result.stderr);print('Receipt:',a.out)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
