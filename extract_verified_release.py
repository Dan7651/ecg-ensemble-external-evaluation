"""Restore the immutable, checksum-verified release into a separate directory."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent
archive=ROOT/'ECG_Ensemble_External_Evaluation_v0.4.1.zip'
expected='8c244b976f6d75166c48fad31707a2056374cb289bf0a5e8427d0e28b9fcab5e'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected,'Public release ZIP SHA256 mismatch'
out=ROOT/'sealed_v0.4.1';out.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
 prefix='ecg-ensemble-external-evaluation/'
 for name in z.namelist():
  assert name.startswith(prefix) and '..' not in Path(name).parts
  if name.endswith('/'):continue
  rel=Path(name.removeprefix(prefix));dest=out/rel;data=z.read(name)
  if dest.exists():assert dest.read_bytes()==data,'Refuse to overwrite differing existing file: '+str(dest)
  else:
   dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.partial');tmp.write_bytes(data);tmp.replace(dest)
 manifest=json.loads((out/'FILE_SHA256.json').read_text())
 for name,digest in manifest.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
print('VERIFIED_SEALED_RELEASE',out,len(manifest),'files; no model/checkpoint deserialization')
