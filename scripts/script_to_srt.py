#!/usr/bin/env python3
"""
Script → draft SRT: split a narration script (.txt / .md) into cue sentences with estimated times.

Purpose: the first step when starting from a topic/script. The draft SRT is only
for sentence splitting; the real timeline is later retimed by
  tts_narration.py --retime-out from actual voice lengths (voice-driven pacing).

Splitting rules:
  - Split at sentence-ending punctuation (. ! ? …); quotes follow their sentence;
  - Sentences over --max-chars are split further at commas/semicolons;
  - Too-short sentences (< --min-chars) merge into the next one;
  - Lines starting with # (Markdown headings) and blank lines are not read; blank lines
    start new paragraphs → longer suggested pause after the paragraph's last cue.
Estimated duration: --cps chars/sec, at least 1.5s.

Usage:
  python script_to_srt.py <script.txt> --output draft.srt [--max-chars 80] [--min-chars 12] [--cps 14]
Last line prints PAUSES=<index=sec,...> (paragraph-final cues), expandable into
tts_narration.py --pause args.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# A sentence = shortest text + sentence-ending punctuation (with trailing quotes/brackets), followed by whitespace or end
_SENTENCE = re.compile(r".+?(?:[.!?…。！？]+[\"”’»)\]]*(?=\s|$)|$)")
_CLAUSE_END = re.compile(r"(?<=[,;:，；：])\s+")


def _fmt(ms: int) -> str:
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _quote_open(text: str) -> bool:
    return text.count("“") > text.count("”") or text.count('"') % 2 == 1


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts, cur = [], ""
    for clause in _CLAUSE_END.split(sentence):
        if cur and len(cur) + 1 + len(clause) > max_chars:
            parts.append(cur)
            cur = clause
        else:
            cur = f"{cur} {clause}".strip()
    if cur:
        parts.append(cur)
    return parts


def split_script(text: str, max_chars: int, min_chars: int) -> list[tuple[str, bool]]:
    """Return [(cue text, is paragraph-final)]."""
    paragraphs, buf = [], []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        paragraphs.append(" ".join(buf))

    cues: list[tuple[str, bool]] = []
    for para in paragraphs:
        pieces: list[str] = []
        pending = ""
        for sent in _SENTENCE.findall(para):
            pending = f"{pending} {sent.strip()}".strip()
            if _quote_open(pending):  # multi-sentence dialogue inside quotes stays together
                continue
            pieces.extend(_split_long(pending, max_chars))
            pending = ""
        if pending:
            pieces.extend(_split_long(pending, max_chars))
        pieces = [p for p in pieces if p]
        merged: list[str] = []
        for p in pieces:
            if merged and len(merged[-1]) < min_chars and len(merged[-1]) + 1 + len(p) <= max_chars:
                merged[-1] = f"{merged[-1]} {p}"
            else:
                merged.append(p)
        cues.extend((m, i == len(merged) - 1) for i, m in enumerate(merged))
    return cues


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Script → draft SRT (split into sentences, estimate times)")
    p.add_argument("script", help="Narration script (.txt / .md)")
    p.add_argument("--output", required=True, help="Draft SRT output path")
    p.add_argument("--max-chars", type=int, default=80, help="Max chars per cue (default 80)")
    p.add_argument("--min-chars", type=int, default=12, help="Shorter sentences merge into the next (default 12)")
    p.add_argument("--cps", type=float, default=14.0, help="Estimated speed: chars/sec (default 14)")
    p.add_argument("--para-pause", type=float, default=0.8, help="Suggested pause after paragraphs, seconds (default 0.8)")
    args = p.parse_args(argv)

    text = Path(args.script).read_text(encoding="utf-8-sig")
    cues = split_script(text, args.max_chars, args.min_chars)
    if not cues:
        print("[err] No readable text in script", file=sys.stderr)
        return 1

    blocks, cursor, pauses = [], 0, []
    for i, (cue, para_end) in enumerate(cues, 1):
        dur = max(1500, round(len(cue) / args.cps * 1000))
        blocks.append(f"{i}\n{_fmt(cursor)} --> {_fmt(cursor + dur)}\n{cue}\n")
        cursor += dur
        if para_end and i < len(cues):
            pauses.append(f"{i}={args.para_pause}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(blocks), encoding="utf-8")

    print(f"Cues: {len(cues)}  estimated: {cursor / 1000:.1f}s", file=sys.stderr)
    for i, (cue, para_end) in enumerate(cues, 1):
        print(f"  {i:>2}{' ¶' if para_end else '  '} {cue}", file=sys.stderr)
    print(f"SRT={out.resolve()}")
    print(f"PAUSES={','.join(pauses)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
