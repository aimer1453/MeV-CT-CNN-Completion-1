#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PDF 文本层双引擎体检（pypdf + PyMuPDF）

用途：回应"PDF 第 N 页是 1 1 1 乱码 / 表格是 HTML"一类审稿质疑时，
      对 main.pdf / main_zh.pdf 做可复现的机器核查，与肉眼渲染互为印证。

检查项（全文 + 指定页）：
  1. 连续重复单数字 token 串（>=5 个），如 "1 1 1 1 1"
  2. HTML 标签（table/tr/td/th/div/span/html/body/p/br）
  3. U+FFFD 替换符 / Unicode 私用区字符 / C0 控制字符
  4. 单行同一 token 重复 >=4 次（区分正常英文虚词与数字垃圾）
  5. 字体嵌入状态（Type0 子集 + ToUnicode ⇒ 提取可靠、无 tofu 风险）
  6. 表格出现页码（核对审稿人说的页码上是否真有表格）
  7. 嵌入图像对象计数（PNG 栅格图的文字物理上不进文本层）

用法：
  PYTHONPATH=<pkgtmp> python scripts/verify_pdf_textlayer.py <pdf> [页号...]
  例：... main.pdf 2 8
输出：JSON 到 stdout；同时把指定页渲染为 PNG（肉眼核查用）到
      runs/_verify_latex/p2_p8/。
"""
import collections
import hashlib
import json
import re
import sys

import pymupdf
from pypdf import PdfReader

DIGIT_RUN = re.compile(r"(?:\b\d\b[ \t]*){5,}")
HTML_TAG = re.compile(r"</?(?:table|tr|td|th|div|span|p|br|html|body|thead|tbody)\b", re.I)
TOKEN_TAGS = ("table", "tr", "td", "th", "div", "span", "p", "br", "html", "body", "thead", "tbody")


def scan_text(text):
    """对一段文本做乱码特征扫描，返回可序列化 dict。"""
    dup = []
    for ln in text.split("\n"):
        toks = ln.split()
        if len(toks) < 4:
            continue
        tok, n = collections.Counter(toks).most_common(1)[0]
        if n >= 4:
            dup.append({"token": tok, "n": n, "line": ln[:70]})
    return {
        "total_chars": len(text),
        "repeated_digit_runs_ge5": len(DIGIT_RUN.findall(text)),
        "repeated_digit_run_samples": DIGIT_RUN.findall(text)[:3],
        "html_tag_count": len(HTML_TAG.findall(text)),
        "html_tag_samples": HTML_TAG.findall(text)[:5],
        "replacement_char_ufffd": text.count("\ufffd"),
        "private_use_chars": sum(1 for c in text if 0xE000 <= ord(c) <= 0xF8FF),
        "control_chars": sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t"),
        "lines_token_repeated_ge4": dup[:5],
        "n_lines_token_repeated_ge4": len(dup),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    src = sys.argv[1]
    pages_want = [int(x) for x in sys.argv[2:]] or [2, 8]

    doc = pymupdf.open(src)
    fitz_pages = [p.get_text("text") for p in doc]
    pypdf_pages = [(p.extract_text() or "") for p in PdfReader(src).pages]

    out = {
        "source": src,
        "page_count": doc.page_count,
        "producer": doc.metadata.get("producer"),
        "engines": {
            "pymupdf": scan_text("\n".join(fitz_pages)),
            "pypdf": scan_text("\n".join(pypdf_pages)),
        },
    }

    # 指定页细查
    page_rep = {}
    for pno in pages_want:
        if not (1 <= pno <= doc.page_count):
            continue
        tf, tp = fitz_pages[pno - 1], pypdf_pages[pno - 1]
        page_rep[f"page_{pno}"] = {
            "pymupdf_chars": len(tf),
            "pypdf_chars": len(tp),
            "pymupdf_head": tf[:160].replace("\n", " | "),
            "pypdf_head": tp[:160].replace("\n", " | "),
            "repeated_digit_runs_pymupdf": len(DIGIT_RUN.findall(tf)),
            "repeated_digit_runs_pypdf": len(DIGIT_RUN.findall(tp)),
            "html_tags_pymupdf": HTML_TAG.findall(tf)[:5],
            "html_tags_pypdf": HTML_TAG.findall(tp)[:5],
        }
        pix = doc[pno - 1].get_pixmap(dpi=200)
        png = f"runs/_verify_latex/p2_p8/{src.split('/')[-1].replace('.pdf','')}_p{pno}.png"
        pix.save(png)
        page_rep[f"page_{pno}"]["render_png"] = png
        page_rep[f"page_{pno}"]["render_md5"] = hashlib.md5(pix.samples).hexdigest()
    out["pages"] = page_rep

    # 表格页码 & 字体 & 图像
    tabs = []
    for i, p in enumerate(doc):
        for m in re.finditer(r"Table\s+\d+", p.get_text("text")):
            tabs.append({"page": i + 1, "label": m.group(0)})
    out["table_mentions"] = tabs

    fonts = {}
    for i in range(doc.page_count):
        for f in doc.get_page_fonts(i, full=True):
            emb = "EMB" if f[1] not in ("n/a", "") else "NOT-EMB"
            fonts.setdefault(f"{emb}|{f[2]}|{f[3]}", set()).add(i + 1)
    out["fonts"] = {k: sorted(v) for k, v in sorted(fonts.items())}

    img_pages = {i + 1: len(doc[i].get_images(full=True)) for i in range(doc.page_count)
                 if doc[i].get_images(full=True)}
    out["embedded_images_by_page"] = img_pages
    out["embedded_images_total"] = sum(img_pages.values())

    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
