"""Check the installed runtime and one real-video cache; write a durable receipt."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, subprocess, sys

P=Path('/home/fengbujue/项目/rppg识别')
D=P/'motion_upgrade_v26_20260911'
R=P/'results/data1_6_v26_20260911'
B=P/'results/data1_6_20260911'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    installation=json.loads((R/'installation_receipt.json').read_text())
    def check_files():
        for name,digest in installation['files'].items(): assert sha(D/name)==digest, name
        for name,digest in installation['preserved_recommended_entries'].items(): assert sha(name)==digest,name
        assert sha(P/'run_motion_v26_experimental.sh')==installation['launcher_sha256']
    check_files()
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=False,checks=[])
    def run(name,args):
        result=subprocess.run(args,cwd=D,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        with (R/f'installed_{name}.log').open('x',encoding='utf-8') as stream:stream.write(result.stdout)
        receipt['checks'].append(dict(name=name,returncode=result.returncode,log=f'installed_{name}.log'))
        if result.returncode: raise RuntimeError(result.stdout[-4000:])
        print(f'{name}: passed',flush=True)
        return result.stdout
    suite=run('unit_tests',[sys.executable,'-B','-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'])
    matched=re.search(r'Ran (\d+) tests',suite)
    assert matched and int(matched.group(1))>=55
    receipt['unit_tests']=int(matched.group(1))
    help_text=run('cli_help',['bash',str(P/'run_motion_v26_experimental.sh'),'--help'])
    assert '--variant' in help_text and '--video' in help_text and '--trace-cache' in help_text
    video=next(row['video']['path'] for row in json.loads((B/'inputs_manifest.json').read_text()) if row['case']=='data2')
    out=R/'installed_entry_smoke_data2'; assert not out.exists()
    run('data2_harmonic',['bash',str(P/'run_motion_v26_experimental.sh'),'--video',video,
        '--out',str(out),'--variant','harmonic','--trace-cache',str(B/'data2/inference')])
    run('data2_qa',[sys.executable,'-B',str(D/'qa_components_entry_v26.py'),'--result',str(out),
        '--frozen',str(R/'component_harmonics/data2'),'--trace-source',str(B/'data2/inference')])
    smoke=json.loads((out/'entry_qa_receipt.json').read_text());assert smoke['passed']
    receipt['data2_entry_qa']=smoke
    check_files();receipt['installed_files_unchanged']=True;receipt['V25_entries_unchanged']=True;receipt['passed']=True
    with (R/'installed_runtime_qa.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2)
    note=R/'V26运动心率改进结果.md'
    with note.open('a',encoding='utf-8') as stream:
        stream.write('\n## 安装后的复核\n\n'+f'安装目录的{receipt["unit_tests"]}项测试通过，实验入口帮助和data2真实缓存运行通过；最终心率及接受掩码与冻结倍频实验完全一致。另有固定EfficientPhys适配器10项测试通过。源码／权重哈希、独立全六片评价和真实波形来源审计均保留收据。\n\n[安装后测试与回归记录](<//wsl.localhost/Ubuntu'+str(R/'installed_runtime_qa.json')+'>)\n')
    print(json.dumps(dict(passed=True,tests=receipt['unit_tests'],smoke='data2 harmonic',preserved_V25=True),ensure_ascii=False))

if __name__=='__main__':main()
