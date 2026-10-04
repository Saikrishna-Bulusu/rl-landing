"""Review copy of the draft: every swappable word highlighted, its alternatives in the highlight's note (pdfcomment).
   cd acra2026 && python3 review/make_review_pdf.py      -> review/main_synonyms.pdf (main.pdf untouched)"""
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); A = os.path.dirname(HERE)
sys.path.insert(0, HERE); from synonyms import S, FIXED, PHRASES
SECTIONS = ["s0_abstract", "s1_introduction", "s2_related", "s3_stack", "s4_method", "s5_results", "s6_gazebo", "s7_discussion"]
PROTECT = re.compile("|".join([r"\\(?:cite|shortcite|ref|label|input|url|textbf|figslot|figslotwide|section|subsection)\*?(?:\{[^{}]*\})+",
                               r"\$[^$]*\$", r"\\[a-zA-Z]+\*?", r"\\."] + [re.escape(p) for p in sorted(PHRASES, key=len, reverse=True)]))
WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
count = {}

def wrap(m):
    w = m.group(0); alts = S.get(w.lower())
    if not alts: return w
    count[w.lower()] = count.get(w.lower(), 0) + 1
    return r"\pdfmarkupcomment[markup=Highlight,color=yellow,author={Alternatives}]{%s}{%s}" % (w, alts)

def process(line):
    if re.match(r"\s*\\(section|subsection|input|figslot|figslotwide|begin|end)\b", line) and not line.lstrip().startswith("\\item"):
        return line
    out, pos = [], 0
    for m in PROTECT.finditer(line):
        out.append(WORD.sub(wrap, line[pos:m.start()])); out.append(m.group(0)); pos = m.end()
    out.append(WORD.sub(wrap, line[pos:]))
    return "".join(out)

for s in SECTIONS:
    src = open(f"{A}/sections/{s}.tex").read().split("\n")
    open(f"{HERE}/sections/{s}.tex", "w").write("\n".join(process(l) for l in src))
main = open(f"{A}/main.tex").read()
main = main.replace(r"\usepackage{acra}", r"\usepackage{acra}" + "\n\\usepackage{pdfcomment}\n\\hypersetup{hidelinks}\n\\hyphenpenalty=10000\\exhyphenpenalty=10000\\sloppy")
main = main.replace(r"\input{sections/", r"\input{review/sections/")
legend = (r"\noindent\fbox{\parbox{0.96\linewidth}{\small\textbf{Review copy.} Highlighted words have alternatives in their notes "
          r"(open the comments panel). Not highlighted on purpose, because each names one defined thing throughout: "
          + ", ".join(FIXED).replace("&", r"\&") + r". Choose one word per meaning and use it everywhere.}}\par\medskip")
main = main.replace(r"\input{review/sections/s0_abstract}", legend + "\n" + r"\input{review/sections/s0_abstract}")
open(f"{HERE}/main_synonyms.tex", "w").write(main)
for cmd in (["pdflatex", "-interaction=nonstopmode", "-output-directory", "review", "review/main_synonyms.tex"],
            ["bibtex", "review/main_synonyms"],
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", "review", "review/main_synonyms.tex"],
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", "review", "review/main_synonyms.tex"],
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", "review", "review/main_synonyms.tex"],
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", "review", "review/main_synonyms.tex"]):
    subprocess.run(cmd, cwd=A, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"{sum(count.values())} highlights over {len(count)} distinct words -> review/main_synonyms.pdf")
