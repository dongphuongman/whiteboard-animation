#!/usr/bin/env python3
"""
URL → kịch bản lời dẫn: tải một trang web (bài báo, blog, wiki),
lọc lấy nội dung chính, gọn thành kịch bản đọc (câu ngắn, đoạn cách
nhau bằng dòng trống) để đưa tiếp vào script_to_srt.py.

Chỉ dùng thư viện chuẩn (urllib + html.parser), không cần key.

Cách lọc: bỏ script/style/nav/header/footer; ưu tiên <article>,
không có thì lấy khối văn bản lớn nhất; bỏ câu quá ngắn (menu,
chú thích) và dòng boilerplate.

Dùng:
  python3 scripts/url_to_script.py <URL> --output script.md [--max-words 400]
  Dòng cuối in SCRIPT=<đường dẫn> và WORDS=<số từ>.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside",
             "form", "button", "select", "noscript", "svg"}
MIN_SENT_CHARS = 15  # câu ngắn hơn coi như menu/chú thích
MIN_CHUNK_WORDS = 40  # đoạn mới khi đủ dài + gặp biên đoạn gốc
MAX_CHUNK_WORDS = 70  # đoạn quá dài thì ngắt (đọc ≤ ~25 giây)
DEFAULT_MAX_WORDS = 400  # ~2 phút đọc tiếng Việt

_ws = re.compile(r"\s+")
_end = re.compile(r"([.!?…]+[\"”']?)\s+")
_cite = re.compile(r"\[\d+(?:[,\s-]*\d+)*\]")  # chú thích kiểu wiki [1], [2, 3]


def normalize_url(url: str) -> str:
    """Encode path/query có dấu để urllib xử lý được."""
    from urllib.parse import quote, urlsplit, urlunsplit

    parts = urlsplit(url.strip())
    return urlunsplit((
        parts.scheme or "https",
        parts.netloc,
        quote(parts.path, safe="/%"),
        quote(parts.query, safe="=&%"),
        parts.fragment,
    ))


def fetch(url: str, timeout: int = 30) -> tuple[str, str]:
    url = normalize_url(url)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (whiteboard-video/1.0)",
        "Accept-Language": "vi,en;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        ctype = resp.headers.get_content_charset() or "utf-8"
    try:
        return raw.decode(ctype, errors="replace"), url
    except LookupError:
        return raw.decode("utf-8", errors="replace"), url


class ArticleParser(HTMLParser):
    """Thu text theo khối <p>/<h2>/li, đánh dấu khối trong <article>."""

    def __init__(self) -> None:
        super().__init__()
        self.skip_depth = 0
        self.in_article = 0
        self.blocks: list[tuple[str, bool]] = []  # (text, trong_article)
        self._buf: list[str] = []
        self._in_block = False
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in SKIP_TAGS:
            self.skip_depth += 1
            return
        if tag == "article":
            self.in_article += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "h1", "h2", "h3", "li") and self.skip_depth == 0:
            self._flush()
            self._in_block = True

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if tag == "article" and self.in_article:
            self.in_article -= 1
        if tag == "title":
            self._in_title = False
        if tag in ("p", "h1", "h2", "h3", "li"):
            self._flush()
            self._in_block = False
        if tag == "br" and self._in_block:
            self._buf.append(" ")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        elif self._in_block:
            self._buf.append(data)

    def _flush(self) -> None:
        text = _ws.sub(" ", "".join(self._buf)).strip()
        text = _cite.sub("", text)
        if text:
            self.blocks.append((html.unescape(text), self.in_article > 0))
        self._buf = []


def split_sentences(text: str) -> list[str]:
    parts = _end.split(text)
    out: list[str] = []
    for i in range(0, len(parts) - 1, 2):
        s = (parts[i] + parts[i + 1]).strip()
        if len(s) >= MIN_SENT_CHARS:
            out.append(s)
    tail = parts[-1].strip() if len(parts) % 2 == 1 else ""
    # Đuôi không có dấu kết câu thường là menu/link sót lại → bỏ
    if len(tail) >= MIN_SENT_CHARS and re.search(r"[.!?…:;][\"”']?$", tail):
        out.append(tail)
    return out


def clean_title(raw: str) -> str:
    t = _ws.sub(" ", raw).strip()
    # Bỏ hậu tố tên site: "Bài viết - VnExpress", "X | Báo Y"
    t = re.split(r"\s[-–|]\s(?=[^-–|]{2,}$)", t)[0].strip()
    return t[:120]


def build_script(title: str, blocks: list[tuple[str, bool]], max_words: int) -> str:
    use_article = any(inside for _, inside in blocks)
    # Gom câu liên tục theo thứ tự, ngắt đoạn ~40-70 từ (biên đoạn gốc = chỗ nghỉ)
    chunks: list[list[str]] = [[]]
    chunk_words = 0
    total = 0
    for text, inside in blocks:
        if use_article and not inside:
            continue
        sents = split_sentences(text)
        for i, s in enumerate(sents):
            w = len(s.split())
            if total + w > max_words and total > 0:
                break
            if chunks[-1] and ((i == 0 and chunk_words >= MIN_CHUNK_WORDS)
                               or chunk_words + w > MAX_CHUNK_WORDS):
                chunks.append([])
                chunk_words = 0
            chunks[-1].append(s)
            chunk_words += w
            total += w
        if total >= max_words:
            break
    chunks = [c for c in chunks if c]
    body = "\n\n".join(" ".join(c) for c in chunks)
    return f"# {title}\n\n{body}\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="URL → kịch bản lời dẫn")
    p.add_argument("url", help="URL bài viết")
    p.add_argument("--output", required=True, help="file script.md đầu ra")
    p.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS,
                   help=f"giới hạn từ (mặc định {DEFAULT_MAX_WORDS} ≈ 2 phút)")
    args = p.parse_args(argv)

    try:
        page, _ = fetch(args.url)
    except Exception as e:  # noqa: BLE001 — lỗi mạng báo gọn cho agent
        print(f"[err] không tải được URL: {e}", file=sys.stderr)
        return 1
    parser = ArticleParser()
    parser.feed(page)
    if not parser.blocks:
        print("[err] không trích được nội dung (trang rỗng hoặc chống bot)",
              file=sys.stderr)
        return 1
    title = clean_title("".join(parser.title_parts)) or "Kịch bản từ URL"
    script = build_script(title, parser.blocks, args.max_words)
    words = len(script.split())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")
    print(f"[ok] {len(parser.blocks)} khối → {words} từ")
    print(f"SCRIPT={out}")
    print(f"WORDS={words}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
