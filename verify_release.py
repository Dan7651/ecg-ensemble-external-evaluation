"""Check the sealed distribution without importing a model or opening a checkpoint."""
from pathlib import Path
import hashlib,json,sys
P=Path(__file__).resolve().parent
m=json.loads((P/'FILE_SHA256.json').read_text())
for name,digest in m.items():
 p=P/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise SystemExit('Integrity failure: '+name)
print('RELEASE_INTEGRITY_PASSED',len(m),'files; no model/checkpoint deserialization')
