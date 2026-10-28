from pathlib import Path
import argparse,shutil,runpy,json,hashlib
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--table');a.add_argument('--figure',choices=['revision_diagnostics']);x=a.parse_args();o=P/'reproduced_revision';o.mkdir(exist_ok=True)
if x.table:
 assert Path(x.table).name==x.table
 f=P/'tables'/(x.table+'.csv');assert f.is_file();pins=json.loads((P/'tables/completion.json').read_text())['outputs'];assert hashlib.sha256(f.read_bytes()).hexdigest()==pins[f.name];shutil.copy2(f,o/f.name);print(o/f.name)
if x.figure:runpy.run_path(str(P/'revision_figures.py'),run_name='__main__')
