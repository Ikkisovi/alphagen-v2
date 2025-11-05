from pathlib import Path
for encoding in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
    try:
        text = Path("lean_project/train_for_lean.py").read_text(encoding=encoding)
        enc = encoding
        break
    except UnicodeDecodeError:
        continue
else:
    raise SystemExit("decode error")
old = "from alphagen.data.expression import Feature, Ref\n"
new = "from alphagen.data.expression import Feature, Ref\nfrom alphagen.data.parser import parse_expression\n"
if old not in text:
    raise SystemExit('import marker missing')
text = text.replace(old, new, 1)
Path("lean_project/train_for_lean.py").write_text(text, encoding=enc)
