#!/usr/bin/env python3
"""
SRT parsing + scene suggestions.

Parse an .srt subtitle file into structured cues and group them into scenes of
~25-35s spoken each, reporting each scene's time range, total duration
(→ sceneDurationMs) and text.

Used as input for step 1 of the whiteboard-video workflow: read narrative
events, plan the illustration strategy, and set sceneDurationMs for each
image's annotation.

Usage:
  python parse_srt.py <subs.srt> [--target-sec 30] [--min-sec 25] [--max-sec 35]

Output: JSON (stdout), fields:
  cues    each cue: {index, startMs, endMs, durMs, text}
  scenes  suggested scenes: {sceneIndex, startMs, endMs, sceneDurationMs, cueRange, text}
Human-readable summary on stderr for direct reading.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


def _to_ms(h: str, m: str, s: str, ms: str) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms.ljust(3, "0"))


def parse_srt(text: str) -> list[dict]:
    """Parse SRT text into a cue list. Tolerates extra blank lines, BOM, comma/dot ms separators."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues: list[dict] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        # 找到含时间轴的行
        time_line_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if time_line_idx is None:
            continue
        times = _TIME.findall(lines[time_line_idx])
        if len(times) < 2:
            continue
        start = _to_ms(*times[0])
        end = _to_ms(*times[1])
        body = " ".join(lines[time_line_idx + 1:]).strip()
        cues.append({
            "index": len(cues) + 1,
            "startMs": start,
            "endMs": end,
            "durMs": max(0, end - start),
            "text": body,
        })
    return cues


def group_scenes(cues: list[dict], target_sec: float, min_sec: float, max_sec: float) -> list[dict]:
    """
    Group consecutive cues into scenes near the target length: cut a scene when
    it reaches target (but no shorter than min, no longer than max — over max
    forces a cut).
    """
    scenes: list[dict] = []
    bucket: list[dict] = []
    target_ms, min_ms, max_ms = target_sec * 1000, min_sec * 1000, max_sec * 1000

    def flush() -> None:
        if not bucket:
            return
        start = bucket[0]["startMs"]
        end = bucket[-1]["endMs"]
        scenes.append({
            "sceneIndex": len(scenes) + 1,
            "startMs": start,
            "endMs": end,
            "sceneDurationMs": max(0, end - start),
            "cueRange": [bucket[0]["index"], bucket[-1]["index"]],
            "text": " ".join(c["text"] for c in bucket).strip(),
        })
        bucket.clear()

    for cue in cues:
        # If adding this cue would exceed max, cut the scene first (avoid overlong scenes)
        if bucket:
            span_with = cue["endMs"] - bucket[0]["startMs"]
            if span_with > max_ms:
                flush()
        bucket.append(cue)
        span = bucket[-1]["endMs"] - bucket[0]["startMs"]
        # Cut once target is reached (and min satisfied)
        if span >= target_ms and span >= min_ms:
            flush()
    flush()
    return scenes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SRT parsing + scene suggestions")
    p.add_argument("srt", help="Subtitle file path (.srt)")
    p.add_argument("--target-sec", type=float, default=30.0, help="Target seconds per scene (default 30)")
    p.add_argument("--min-sec", type=float, default=25.0, help="Min seconds per scene (default 25)")
    p.add_argument("--max-sec", type=float, default=35.0, help="Max seconds per scene (default 35)")
    args = p.parse_args(argv)

    try:
        raw = Path(args.srt).read_text(encoding="utf-8-sig")
    except OSError as e:
        print(f"[err] Cannot read subtitles: {e}", file=sys.stderr)
        return 1

    cues = parse_srt(raw)
    if not cues:
        print("[err] No cues parsed, check SRT format", file=sys.stderr)
        return 1
    scenes = group_scenes(cues, args.target_sec, args.min_sec, args.max_sec)

    total_ms = cues[-1]["endMs"] - cues[0]["startMs"]
    print(f"Cues: {len(cues)}  total: {total_ms/1000:.1f}s  suggested scenes: {len(scenes)}", file=sys.stderr)
    for s in scenes:
        print(f"  scene{s['sceneIndex']:>2}  {s['startMs']/1000:6.1f}-{s['endMs']/1000:6.1f}s "
              f"({s['sceneDurationMs']/1000:4.1f}s, cues {s['cueRange'][0]}-{s['cueRange'][1]}): "
              f"{s['text'][:40]}", file=sys.stderr)

    json.dump({"cues": cues, "scenes": scenes}, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
