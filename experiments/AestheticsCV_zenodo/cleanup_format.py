"""
cleanup_format.py -- mechanical de-decoration of manuscript.tex.
Strips \\textbf, \\emph (outside the bibliography), \\paragraph and \\subsection;
merges the sentence-titled sections into plain Introduction / Methods / Results /
Discussion / Limitations / Conclusion; converts every LaTeX dash (--- and --) to a
hyphen or removes table N/A markers. Prose text is preserved verbatim.
"""
import re

PATH = "manuscript.tex"
with open(PATH, encoding="utf-8") as f:
    s = f.read()

# ---- 1. section renames / merges (strings still contain LaTeX '--') ----
s = s.replace(r"\section{A taxonomy of nostalgic aesthetics}", r"\section{Methods}")
for sec in (
    r"\section{Method: a measurable affective--temporal space}",
    r"\section{Frontier vision--language models read the structure and over-attribute unease}",
    r"\section{The over-attribution operates at the level of interpretation}",
):
    s = s.replace(sec + "\n", "")
    s = s.replace(sec, "")

# ---- 2. remove \subsection{...} entirely ----
n_sub = len(re.findall(r"\\subsection\{[^}]*\}", s))
s = re.sub(r"\\subsection\{[^}]*\}[ \t]*\n?", "", s)

# ---- 3. unwrap \paragraph{X} -> X (plain topic lead-in, no bold) ----
n_par = len(re.findall(r"\\paragraph\{[^}]*\}", s))
s = re.sub(r"\\paragraph\{([^}]*)\}", r"\1", s)

# ---- 4. strip \textbf / \textit everywhere ----
for _ in range(4):
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^{}]*)\}", r"\1", s)

# ---- 5. strip \emph{...} EXCEPT inside thebibliography (keep journal-title italics) ----
m = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", s, re.DOTALL)
bib = None
if m:
    bib = m.group(0)
    s = s[:m.start()] + "@@BIBBLOCK@@" + s[m.end():]
for _ in range(4):
    s = re.sub(r"\\emph\{([^{}]*)\}", r"\1", s)
if bib is not None:
    s = s.replace("@@BIBBLOCK@@", bib)

# ---- 6. dashes ----
# 6a. table N/A cells: '---' immediately followed by '&' -> remove
s = re.sub(r"---(?=\s*&)", "", s)
# 6b. documented N/A caption marker
s = s.replace("(``---'' otherwise, by design)", "(blank otherwise, by design)")
# 6c. en/em dashes -> hyphen everywhere (vision--language -> vision-language, etc.)
s = s.replace("---", "-").replace("--", "-")

with open(PATH, "w", encoding="utf-8") as f:
    f.write(s)

print(f"subsections removed : {n_sub}")
print(f"paragraphs unwrapped: {n_par}")
print(f"textbf remaining    : {len(re.findall(chr(92)+chr(92)+'textbf', s))}")
print(f"emph remaining (bib) : {len(re.findall(chr(92)+chr(92)+'emph', s))}")
print(f"'---' remaining      : {s.count('---')}")
print(f"'--' remaining       : {s.count('--')}")
