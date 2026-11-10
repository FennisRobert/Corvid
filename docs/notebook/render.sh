#!/bin/sh
# Compile progress.typ to PDF and to page PNGs for the published notebook page.
# Needs typst and the fonts in fonts/ (EB Garamond and Garamond-Math, SIL Open Font Licence; not in the repository).
cd "$(dirname "$0")" || exit 1
OUT="$(pwd)/notebook"
mkdir -p "$OUT"
typst compile --font-path fonts progress.typ progress.pdf || exit 1
rm -f "$OUT"/p[0-9]*.png
typst compile --font-path fonts --format png --ppi 150 progress.typ "$OUT/p{p}.png" || exit 1
ls "$OUT" | grep -c '^p[0-9]*\.png'
# Regenerate the page wrapper for the current page count.
N=$(ls "$OUT" | grep -c '^p[0-9]*\.png')
STAMP=$(date "+%-d %B %Y, %H:%M")
python3 - "$OUT" "$N" "$STAMP" <<'PY'
import sys, re
out, n, stamp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
path = f"{out}/corvid-notebook.html"
s = open(path).read()
head = s[:s.index('  <figure>')]
head = re.sub(r'updated [^<]*', f'updated {stamp}', head)
figs = "".join(f'  <figure><img class="sheet" src="p{i}.png" alt="Notebook page {i}"><figcaption>page {i} of {n}</figcaption></figure>\n' for i in range(1, n + 1))
open(path, "w").write(head + figs + "</div>\n")
PY
