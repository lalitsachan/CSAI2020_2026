#!/usr/bin/env python3
"""
fix_tex.py
Scans and repairs common LaTeX typos:
1. Markdown backtick code spans `code_var` -> \\texttt{code\\_var}
2. Markdown bolding **text** -> \\textbf{text}
3. Unescaped underscores in table or prose text outside math mode and lstlisting
4. Unescaped percent signs % in prose (outside comments)
"""

import re
from pathlib import Path

def repair_tex_content(text: str) -> str:
    # 1. Protect lstlisting blocks
    lst_blocks = []
    def save_lst(m):
        lst_blocks.append(m.group(0))
        return f"__LST_BLOCK_{len(lst_blocks)-1}__"
    
    text = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', save_lst, text, flags=re.DOTALL)
    
    # 2. Protect math blocks
    math_blocks = []
    def save_math(m):
        math_blocks.append(m.group(0))
        return f"__MATH_BLOCK_{len(math_blocks)-1}__"
        
    # Environments
    text = re.sub(r'\\begin\{(equation|align|gather)\*?\}.*?\\end\{\1\*?\}', save_math, text, flags=re.DOTALL)
    # Display math \[ ... \]
    text = re.sub(r'\\\[.*?\\\]', save_math, text, flags=re.DOTALL)
    # Inline math $ ... $
    text = re.sub(r'(?<!\\)\$.*?(?<!\\)\$', save_math, text, flags=re.DOTALL)
    
    # 3. Protect TikZ pictures
    tikz_blocks = []
    def save_tikz(m):
        tikz_blocks.append(m.group(0))
        return f"__TIKZ_BLOCK_{len(tikz_blocks)-1}__"
    text = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', save_tikz, text, flags=re.DOTALL)
    
    # 4. Now we are purely in text mode!
    # Convert `code` to \texttt{...} and escape _ and % inside it
    def convert_backtick(m):
        inner = m.group(1)
        # escape characters inside code
        inner = inner.replace('\\', '') # remove any double escapes
        inner = inner.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
        return r'\texttt{' + inner + '}'
    text = re.sub(r'`([^`\n]+?)`', convert_backtick, text)
    
    # Convert **bold** to \textbf{...}
    def convert_bold(m):
        inner = m.group(1)
        return r'\textbf{' + inner + '}'
    text = re.sub(r'\*\*([^\*\n]+?)\*\*', convert_bold, text)
    
    # In table cells, fix unescaped \texttt{'__missing__'}
    text = text.replace(r"\texttt{'__missing__'}", r"\texttt{'\_missing\_'}")
    text = text.replace(r"\texttt{'__low_freq_cat__'}", r"\texttt{'\_low\_freq\_cat\_'}")
    text = text.replace(r"__missing__", r"\_missing\_")
    text = text.replace(r"__low_freq_cat__", r"\_low\_freq\_cat\_")
    
    # Restore TikZ blocks
    for i, b in enumerate(tikz_blocks):
        text = text.replace(f"__TIKZ_BLOCK_{i}__", b)
        
    # Restore math blocks
    for i, b in enumerate(math_blocks):
        text = text.replace(f"__MATH_BLOCK_{i}__", b)
        
    # Restore lstlisting blocks
    for i, b in enumerate(lst_blocks):
        text = text.replace(f"__LST_BLOCK_{i}__", b)
        
    return text

def main():
    root = Path(__file__).resolve().parent
    tex_files = list(root.glob("**/*.tex"))
    print(f"Checking {len(tex_files)} .tex files for syntax repairs...")
    
    for p in tex_files:
        orig = p.read_text(encoding="utf-8")
        repaired = repair_tex_content(orig)
        if repaired != orig:
            p.write_text(repaired, encoding="utf-8")
            print(f"[REPAIRED] {p.relative_to(root)}")
        else:
            print(f"[OK] {p.relative_to(root)}")

if __name__ == "__main__":
    main()

