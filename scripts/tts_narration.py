#!/usr/bin/env python3
"""
SRT narration dubbing: synthesize each SRT cue to speech, lay it on the subtitle
timeline as one narration track, optionally mux straight into the final MP4
(video stream not re-encoded).

Three engines (--provider, default from .env TTS_PROVIDER):
  vbee  POST /api/v1/tts → request_id → poll GET /api/v1/tts/{id} → download audio_link (mp3)
         Needs VBEE_APP_ID / VBEE_ACCESS_TOKEN, billed per character.
  edge  rany2/edge-tts (free online Microsoft Edge read-aloud), default voice vi-VN-NamMinhNeural.
  vietneu  VieNeu-TTS on-device (ONNX/CPU, offline after the first ~230MB model download),
         default voice Minh Đức (male · northern · news), change preset with --voice.
Synthesis results are cached by engine+text+voice+speed; reruns never re-request.
A cue longer than its slot (up to the next cue's start) is squeezed in with atempo + warning.

Defaults from project-root .env / env vars:
  TTS_PROVIDER, TTS_VOICE (Vbee voice_code passed as-is, or Edge voice name), TTS_SPEED (1.0 = normal)

Usage:
  <ENV_PY> tts_narration.py <subs.srt> --output narration.m4a [--video final.mp4 --video-out final-voice.mp4]
                             [--provider vbee|edge|vietneu] [--voice <voice>] [--speed 1.1] [--cache-dir <dir>]
                             [--retime-out tight.srt --gap 0.3 --pause 6=0.8 --tail 1.0] [--no-trim]
  --retime-out: rebuild the timeline from real voice lengths (drop gaps between cues),
                then sync annotations with retime_annotations.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_srt import parse_srt  # noqa: E402

API_URL = "https://vbee.vn/api/v1/tts"
CALLBACK_URL = "https://example.com/callback"  # API requires it; placeholder under polling mode
POLL_INTERVAL_S = 2
POLL_MAX = 30
SAMPLE_RATE = 44100

VBEE_DEFAULT_VOICE = "n_hanoi_male_protrainer_education_vc"
EDGE_DEFAULT_VOICE = "vi-VN-NamMinhNeural"
EDGE_COMMA_TRIES = 4  # after the raw sentence fails, try variants with added commas
VIETNEU_DEFAULT_VOICE = "Minh Đức"  # Male · Northern · news
VIETNEU_BACKBONE_REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
VIETNEU_BACKBONE_REV = "2da0efab622a1722125991736524f080b751ef5b"
VIETNEU_BACKBONE_FILES = ["config.json", "denoiser.onnx", "speaker_encoder.onnx",
                          "onnx_int8/config.json", "onnx_int8/tokenizer.json",
                          "onnx_int8/vieneu_acoustic_cached.onnx",
                          "onnx_int8/vieneu_backbone_shared.data",
                          "onnx_int8/vieneu_prefill.onnx",
                          "onnx_int8/vieneu_decode_step.onnx",
                          "onnx_int8/vieneu_v3_heads.npz"]


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE, skip comments; existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _request_json(url: str, token: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method="POST" if body is not None else "GET",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "whiteboard-video/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def synthesize(text: str, app_id: str, token: str, voice: str, speed: float) -> str:
    """Submit synthesis and poll; return audio_link."""
    data = _request_json(API_URL, token, {
        "app_id": app_id,
        "input_text": text,
        "voice_code": voice,
        "audio_type": "mp3",
        "speed_rate": speed,
        "callback_url": CALLBACK_URL,
    })
    if data.get("status") != 1:
        raise RuntimeError(f"Vbee error: {data.get('error_message') or data.get('error_code')}")
    result = data.get("result") or {}
    if result.get("audio_link"):
        return result["audio_link"]
    request_id = result.get("request_id")
    if not request_id:
        raise RuntimeError("Vbee returned no request_id")

    for _ in range(POLL_MAX):
        time.sleep(POLL_INTERVAL_S)
        try:
            status = _request_json(f"{API_URL}/{request_id}", token)
        except urllib.error.URLError:
            continue
        if status.get("status") != 1:
            continue
        res = status.get("result") or {}
        if res.get("status") == "SUCCESS" and res.get("audio_link"):
            return res["audio_link"]
        if res.get("status") == "FAILURE":
            raise RuntimeError(f"Vbee synthesis failed: request_id={request_id}")
    raise RuntimeError(f"Vbee poll timeout ({POLL_INTERVAL_S * POLL_MAX}s): request_id={request_id}")


def _edge_variants(text: str) -> list[str]:
    """Raw sentence x2, plus variants with a comma inserted near mid-sentence (middle outward)."""
    words = text.split(" ")
    mid = len(words) // 2
    order = sorted(range(1, len(words)), key=lambda k: abs(k - mid))
    commas = [" ".join(words[:k]) + ", " + " ".join(words[k:]) for k in order
              if not words[k - 1].endswith((",", ".", "!", "?", ":", ";"))]
    return [text, text] + commas[:EDGE_COMMA_TRIES]


def synthesize_edge(text: str, voice: str, speed: float, out: Path) -> None:
    """edge-tts synthesizes at base speed, then ffmpeg atempo adjusts speed (pitch kept).

    Edge servers return NoAudioReceived for some Vietnamese sentences (more common
    with rate set); unrelated to network, retries don't help, but adding one comma
    mid-sentence fixes it. So: no rate sent; retry raw sentence twice, then try
    comma-inserted variants near the middle (one extra slight pause, same content).
    """
    import edge_tts

    raw = out.with_suffix(".raw.mp3")
    for i, variant in enumerate(_edge_variants(text)):
        try:
            edge_tts.Communicate(variant, voice).save_sync(str(raw))
        except edge_tts.exceptions.NoAudioReceived:
            time.sleep(1)
            continue
        if variant != text:
            print(f"  [warn] Edge cannot synthesize raw sentence, using: {variant}")
        break
    else:
        raise RuntimeError(f"Edge returned NoAudioReceived repeatedly: {text}")
    if abs(speed - 1.0) < 1e-3:
        raw.replace(out)
        return
    tmp = out.with_suffix(".part.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-af", _atempo_chain(speed),
         "-c:a", "libmp3lame", "-q:a", "2", str(tmp)],
        check=True,
    )
    raw.unlink(missing_ok=True)
    tmp.replace(out)


def download(url: str, out: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "whiteboard-video/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        tmp = out.with_suffix(".part")
        tmp.write_bytes(resp.read())
        tmp.replace(out)


_VIETNEU_ENGINE = None


def _vietneu_model_dir() -> Path:
    custom = os.environ.get("VIETNEU_MODEL_DIR", "").strip()
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".cache" / "whiteboard-video" / "vieneu" / "backbone"


def _ensure_vietneu_models(model_dir: Path) -> None:
    """Lần đầu tải model int8 (~230MB) về thư mục phẳng; các lần sau dùng lại, offline."""
    marker = model_dir / "onnx_int8" / "vieneu_prefill.onnx"
    if marker.exists():
        return
    print("[..] Downloading VietNeu models (one time only, ~230MB)...")
    from huggingface_hub import snapshot_download

    snapshot_download(VIETNEU_BACKBONE_REPO, revision=VIETNEU_BACKBONE_REV,
                      local_dir=str(model_dir), allow_patterns=VIETNEU_BACKBONE_FILES)
    if not marker.exists():
        raise RuntimeError("VietNeu model download failed")


def synthesize_vietneu(text: str, voice: str, speed: float, out: Path) -> None:
    """VietNeu on-device (ONNX/CPU, offline after the first model download).
    Speed via ffmpeg atempo (pitch kept), same handling as Edge."""
    global _VIETNEU_ENGINE
    if _VIETNEU_ENGINE is None:
        try:
            from vieneu import Vieneu
        except ImportError:
            raise RuntimeError("missing package vieneu, run: .venv/bin/python -m pip install vieneu")
        model_dir = _vietneu_model_dir()
        _ensure_vietneu_models(model_dir)
        precision = (os.environ.get("VIETNEU_PRECISION") or "int8").lower()
        _VIETNEU_ENGINE = Vieneu(backbone_repo=str(model_dir),
                                onnx_dir=str(model_dir / "onnx_int8"),
                                precision=precision)
    raw = out.with_suffix(".raw.wav")
    _VIETNEU_ENGINE.save(_VIETNEU_ENGINE.infer(text, voice=voice), str(raw))
    af = "" if abs(speed - 1.0) < 1e-3 else f"-af {_atempo_chain(speed)} "
    tmp = out.with_suffix(".part.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), *af.split(),
         "-c:a", "libmp3lame", "-q:a", "2", str(tmp)],
        check=True,
    )
    raw.unlink(missing_ok=True)
    tmp.replace(out)


def probe_duration(path: Path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(res.stdout.strip())


def trim_silence(clip: Path) -> Path:
    """Trim head/tail silence of a TTS clip (Edge ~0.25s head + 0.8s tail), keep 50ms/100ms margins; cached as .trim.wav."""
    out = clip.with_suffix(".trim.wav")
    if not out.exists():
        sr = "silenceremove=start_periods=1:start_threshold=-40dB:start_silence={}"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
             "-af", f"{sr.format(0.05)},areverse,{sr.format(0.1)},areverse",
             "-ar", str(SAMPLE_RATE), "-ac", "1", str(out)],
            check=True,
        )
    return out


def _atempo_chain(factor: float) -> str:
    """atempo single-stage range 0.5–2.0, chain beyond that."""
    parts = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor /= 0.5
    parts.append(f"atempo={factor:.4f}")
    return ",".join(parts)


def build_track(cues: list[dict], clips: list[Path], output: Path, total_ms: int | None) -> None:
    """Place each voice into its [cue start, next cue start) slot: speed up if overlong, pad silence if short, then concat."""
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    first_start = cues[0]["startMs"]
    if first_start > 0:
        filters.append(f"aevalsrc=0:d={first_start / 1000:.3f}:s={SAMPLE_RATE}[lead]")
        labels.append("[lead]")

    for i, (cue, clip) in enumerate(zip(cues, clips)):
        if i + 1 < len(cues):
            slot_ms = cues[i + 1]["startMs"] - cue["startMs"]
        else:
            slot_ms = max(cue["endMs"], total_ms or 0) - cue["startMs"]
        slot_s = slot_ms / 1000
        dur = probe_duration(clip)
        chain = f"[{i}:a]aresample={SAMPLE_RATE},aformat=channel_layouts=mono"
        if dur > slot_s - 0.05:
            factor = dur / (slot_s - 0.05)
            print(f"  [warn] Cue {cue['index']} voice {dur:.2f}s exceeds slot {slot_s:.2f}s, speeding up x{factor:.2f}")
            chain += "," + _atempo_chain(factor)
        chain += f",apad,atrim=0:{slot_s:.3f}[c{i}]"
        inputs += ["-i", str(clip)]
        filters.append(chain)
        labels.append(f"[c{i}]")

    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs,
         "-filter_complex", ";".join(filters), "-map", "[out]",
         "-c:a", "aac", "-b:a", "192k", str(output)],
        check=True,
    )


def retime_cues(cues: list[dict], clips: list[Path], gap_s: float, pauses: dict[int, float],
                tail_s: float) -> list[dict]:
    """Rebuild timeline from real voice lengths: each cue follows the previous + gap (longer pause after specified cues), tail after last."""
    out: list[dict] = []
    cursor = cues[0]["startMs"]
    for i, (cue, clip) in enumerate(zip(cues, clips)):
        dur_ms = round(probe_duration(clip) * 1000)
        end = cursor + dur_ms
        is_last = i + 1 == len(cues)
        hold_ms = round((tail_s if is_last else pauses.get(cue["index"], gap_s)) * 1000)
        out.append({**cue, "startMs": cursor, "endMs": end + (hold_ms if is_last else 0),
                    "durMs": dur_ms})
        cursor = end + hold_ms
    return out


def _fmt_srt_time(ms: int) -> str:
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[dict], path: Path) -> None:
    blocks = [f"{c['index']}\n{_fmt_srt_time(c['startMs'])} --> {_fmt_srt_time(c['endMs'])}\n{c['text']}\n"
              for c in cues]
    path.write_text("\n".join(blocks), encoding="utf-8")


def mux(video: Path, audio: Path, output: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-t", f"{probe_duration(video):.3f}", "-movflags", "+faststart", str(output)],
        check=True,
    )


def main(argv=None) -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    p = argparse.ArgumentParser(description="SRT → narration track (Vbee / Edge / VietNeu, muxable into final)")
    p.add_argument("srt", help="Subtitle file (.srt)")
    p.add_argument("--output", required=True, help="Narration output (.m4a)")
    p.add_argument("--video", help="Final MP4 to mux narration into")
    p.add_argument("--video-out", help="Narrated output path (default <video>-voice.mp4)")
    p.add_argument("--provider", choices=["vbee", "edge", "vietneu"],
                   default=(os.environ.get("TTS_PROVIDER") or "vbee").lower(), help="TTS engine")
    p.add_argument("--voice", help="Voice: Vbee voice_code (copied from Vbee UI), Edge voice name (e.g. vi-VN-HoaiMyNeural), or VietNeu preset (e.g. Minh Đức / Hải Đăng / Mai Anh)")
    p.add_argument("--speed", type=float, default=float(os.environ.get("TTS_SPEED") or 1.0),
                   help="Speed 0.1–1.9 (1.0 = normal)")
    p.add_argument("--cache-dir", help="Per-cue voice cache dir (default <srt dir>/tts-cache)")
    p.add_argument("--retime-out", help="Rebuild timeline from real voice lengths, write compact SRT (track follows new timeline)")
    p.add_argument("--gap", type=float, default=0.3, help="Gap between cues when retiming, seconds (default 0.3)")
    p.add_argument("--pause", action="append", default=[], metavar="INDEX=SEC",
                   help="Longer pause after a cue, repeatable, e.g. --pause 6=0.8")
    p.add_argument("--tail", type=float, default=1.0, help="Hold after last cue when retiming, seconds (default 1.0)")
    p.add_argument("--no-trim", action="store_true", help="Don't trim head/tail silence of each voice")
    args = p.parse_args(argv)
    try:
        pauses = {int(k): float(v) for k, v in (x.split("=", 1) for x in args.pause)}
    except ValueError:
        print("[err] --pause format is INDEX=SEC, e.g. 6=0.8", file=sys.stderr)
        return 1

    env_voice = os.environ.get("TTS_VOICE") or ""
    if args.provider == "vbee":
        app_id = os.environ.get("VBEE_APP_ID")
        token = os.environ.get("VBEE_ACCESS_TOKEN")
        if not app_id or not token:
            print("[err] Missing VBEE_APP_ID / VBEE_ACCESS_TOKEN (.env or env vars)", file=sys.stderr)
            return 1
        # voice_code passed as-is to the API; .env TTS_VOICE in Edge form is not reused
        voice = args.voice or (env_voice if env_voice and not env_voice.endswith("Neural")
                               else VBEE_DEFAULT_VOICE)
    elif args.provider == "vietneu":
        # offline on-device; --voice is a preset name (e.g. Hải Đăng, Mai Anh)
        voice = args.voice or os.environ.get("VIETNEU_VOICE") or VIETNEU_DEFAULT_VOICE
    else:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            print("[err] Missing edge-tts, run prepare_env.py first", file=sys.stderr)
            return 1
        # .env TTS_VOICE may be a Vbee code; reuse only Edge-form voice names
        voice = args.voice or (env_voice if env_voice.endswith("Neural") else EDGE_DEFAULT_VOICE)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("[err] System ffmpeg / ffprobe required", file=sys.stderr)
        return 1
    if not 0.1 <= args.speed <= 1.9:
        print("[err] --speed must be 0.1–1.9", file=sys.stderr)
        return 1

    srt = Path(args.srt)
    cues = [c for c in parse_srt(srt.read_text(encoding="utf-8-sig")) if c["text"]]
    if not cues:
        print("[err] No cues parsed", file=sys.stderr)
        return 1

    cache = Path(args.cache_dir) if args.cache_dir else srt.parent / "tts-cache"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"{args.provider} narration: {len(cues)} cues, voice {voice}, speed {args.speed}")

    clips: list[Path] = []
    for cue in cues:
        key_src = f"{args.provider}|{voice}|{args.speed}|{cue['text']}"
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:12]
        clip = cache / f"cue-{cue['index']:03d}-{key}.mp3"
        if not clip.exists():
            try:
                if args.provider == "vbee":
                    download(synthesize(cue["text"], app_id, token, voice, args.speed), clip)
                elif args.provider == "vietneu":
                    synthesize_vietneu(cue["text"], voice, args.speed, clip)
                else:
                    synthesize_edge(cue["text"], voice, args.speed, clip)
            except Exception as e:  # 网络/服务端错误统一报告并退出
                print(f"[err] 字幕 {cue['index']} 合成失败: {e}", file=sys.stderr)
                return 1
            print(f"  字幕 {cue['index']:>2} 合成完成 ({probe_duration(clip):.2f}s): {cue['text'][:40]}")
        else:
            print(f"  字幕 {cue['index']:>2} 使用缓存")
        clips.append(clip if args.no_trim else trim_silence(clip))

    if args.retime_out:
        cues = retime_cues(cues, clips, args.gap, pauses, args.tail)
        retimed = Path(args.retime_out)
        write_srt(cues, retimed)
        print(f"  时间轴重排: {cues[-1]['endMs'] / 1000:.1f}s")
        print(f"SRT={retimed.resolve()}")

    video = Path(args.video) if args.video else None
    total_ms = int(probe_duration(video) * 1000) if video else None
    output = Path(args.output)
    build_track(cues, clips, output, total_ms)
    print(f"AUDIO={output.resolve()}")

    if video:
        video_out = Path(args.video_out) if args.video_out else video.with_name(f"{video.stem}-voice.mp4")
        mux(video, output, video_out)
        print(f"OUTPUT={video_out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
