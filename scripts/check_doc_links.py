"""Validate local documentation links and heading anchors without network access."""
import re
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
files=[ROOT/'README.md',ROOT/'experiment_results.md',*sorted((ROOT/'docs').rglob('*.md'))]
count=0
for path in files:
    body=path.read_text(encoding='utf-8')
    for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)',body):
        if '://' in link or link.startswith('mailto:'):continue
        filename,_,anchor=unquote(link).partition('#')
        target=(path.parent/filename).resolve() if filename else path
        assert target.exists(),f'{path.relative_to(ROOT)}: missing {link}'
        if anchor:
            assert target.suffix=='.md',link
            headings=re.findall(r'^#{1,6}\s+(.+)$',target.read_text(encoding='utf-8'),flags=re.M)
            slugs=[re.sub(r'[^\w\- ]','',heading.lower()).replace(' ','-') for heading in headings]
            assert anchor in slugs,f'{path.relative_to(ROOT)}: missing anchor {link}'
        count+=1
print(f'PASS: {count} local links/anchors across {len(files)} documentation files.')
