#!/usr/bin/env python3
"""
Tự sinh ảnh nét vẽ cho các cảnh: đọc IMAGE_PROMPTS.md trong thư mục dự án,
gọi API tạo ảnh, lưu đúng tên file PNG (tỉ lệ 16:9).

Hỗ trợ 3 engine (--provider, mặc định đọc .env IMG_PROVIDER):
  gemini  Google AI Studio (generateContent + responseModalities IMAGE),
          mặc định model gemini-2.5-flash-image. Cần GEMINI_API_KEY.
  openai  gpt-image-1, size 1536x1024. Cần OPENAI_API_KEY.
  agnes   Agnes Image 2.5 Flash (POST apihub.agnes-ai.com/v1/images/generations,
          size 1K + ratio 16:9, hiện miễn phí). Cần AGNES_API_KEY.

Ảnh tải về được chuẩn hoá 16:9 bằng Pillow (ImageOps.fit, cắt giữa nếu lệch).
File đã tồn tại thì bỏ qua (dùng --force để tạo lại).
Chỉ dùng thư viện chuẩn + Pillow (đã có trong .venv).

Đọc key từ <repo>/.env (GEMINI_API_KEY / OPENAI_API_KEY / IMG_PROVIDER / IMG_MODEL).

Dùng IMAGE_PROMPTS.md do bước 2 tạo ra, định dạng mỗi mục:
    ```prompt...
    ```
    Lưu thành: `scene-01-ten.png`

Dùng:
  <ENV_PY> scripts/generate_images.py assets/whiteboard/<du-an>
      [--provider gemini|openai] [--model <ten-model>] [--force]
      [--only scene-01-ten.png]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-image"
OPENAI_DEFAULT_MODEL = "gpt-image-1"
AGNES_API_URL = "https://apihub.agnes-ai.com/v1/images/generations"
AGNES_DEFAULT_MODEL = "agnes-image-2.5-flash"
OUT_W, OUT_H = 1536, 864  # 16:9 chuẩn hoá đầu ra


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_prompts(md_path: Path) -> list[tuple[str, str]]:
    """Trích (ten_file, prompt) từ IMAGE_PROMPTS.md."""
    text = md_path.read_text(encoding="utf-8")
    items: list[tuple[str, str]] = []
    # Mỗi mục: một khối ``` ... ``` rồi dòng "Lưu thành: `xxx.png`"
    for m in re.finditer(r"```\s*\n(.*?)```[^\n]*\n+[^\n]*Lưu thành:\s*`([^`]+)`", text, re.S):
        prompt, filename = m.group(1).strip(), m.group(2).strip()
        if prompt and filename.endswith(".png"):
            items.append((filename, prompt))
    return items


def _post_json(url: str, body: dict, headers: dict | None = None, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_bytes(url: str, headers: dict | None = None, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


STYLE_SUFFIX = (
    " Wide 16:9 landscape composition. Flat warm cream paper background,"
    " minimal dark-gray hand sketch, absolutely no text, numbers, letters or logos."
)


def gen_gemini(prompt: str, model: str, api_key: str) -> bytes:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    data = _post_json(
        url,
        {"contents": [{"parts": [{"text": prompt + STYLE_SUFFIX}]}],
         "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}},
        headers={"x-goog-api-key": api_key},
    )
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini trả về bất thường: {json.dumps(data)[:300]}")
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return base64.b64decode(inline["data"])
    raise RuntimeError(f"Gemini không trả ảnh: {json.dumps(data)[:300]}")


def gen_openai(prompt: str, model: str, api_key: str) -> bytes:
    data = _post_json(
        "https://api.openai.com/v1/images/generations",
        {"model": model, "prompt": prompt + STYLE_SUFFIX, "size": "1536x1024"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        item = data["data"][0]
    except (KeyError, IndexError):
        raise RuntimeError(f"OpenAI trả về bất thường: {json.dumps(data)[:300]}")
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    if item.get("url"):
        return _get_bytes(item["url"])
    raise RuntimeError(f"OpenAI không trả ảnh: {json.dumps(data)[:300]}")


def gen_agnes(prompt: str, model: str, api_key: str) -> bytes:
    """Agnes Image: size tier + ratio 16:9, ưu tiên Base64 (theo docs)."""
    data = _post_json(
        AGNES_API_URL,
        {"model": model, "prompt": prompt + STYLE_SUFFIX,
         "size": "1K", "ratio": "16:9", "return_base64": True},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=300,
    )
    try:
        item = data["data"][0]
    except (KeyError, IndexError):
        raise RuntimeError(f"Agnes trả về bất thường: {json.dumps(data)[:300]}")
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    if item.get("url"):
        return _get_bytes(item["url"])
    raise RuntimeError(f"Agnes không trả ảnh: {json.dumps(data)[:300]}")


def save_16x9(raw: bytes, out_path: Path) -> tuple[int, int]:
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = ImageOps.fit(img, (OUT_W, OUT_H), method=Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return img.size


def main(argv=None) -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    p = argparse.ArgumentParser(description="Tự sinh ảnh nét vẽ từ IMAGE_PROMPTS.md")
    p.add_argument("project", help="thư mục dự án chứa IMAGE_PROMPTS.md")
    p.add_argument("--provider", choices=["gemini", "openai", "agnes"],
                   default=(os.environ.get("IMG_PROVIDER") or "gemini").lower())
    p.add_argument("--model", default=os.environ.get("IMG_MODEL") or "")
    p.add_argument("--force", action="store_true", help="tạo lại cả ảnh đã có")
    p.add_argument("--only", action="append", default=[],
                   help="chỉ tạo file này (dùng được nhiều lần)")
    args = p.parse_args(argv)

    proj = Path(args.project)
    md = proj / "IMAGE_PROMPTS.md"
    if not md.exists():
        print(f"[err] không thấy {md}", file=sys.stderr)
        return 1
    items = parse_prompts(md)
    if args.only:
        items = [(f, pr) for f, pr in items if f in args.only]
    if not items:
        print("[err] không trích được prompt nào (xem lại định dạng IMAGE_PROMPTS.md)",
              file=sys.stderr)
        return 1

    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        model = args.model or GEMINI_DEFAULT_MODEL
        gen = gen_gemini
        var = "GEMINI_API_KEY"
    elif args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = args.model or OPENAI_DEFAULT_MODEL
        gen = gen_openai
        var = "OPENAI_API_KEY"
    else:
        api_key = os.environ.get("AGNES_API_KEY", "")
        model = args.model or AGNES_DEFAULT_MODEL
        gen = gen_agnes
        var = "AGNES_API_KEY"
    if not api_key:
        print(f"[err] thiếu {var} — thêm vào .env rồi chạy lại", file=sys.stderr)
        return 1

    print(f"[..] {len(items)} ảnh, provider={args.provider}, model={model}")
    ok, skip = 0, 0
    for filename, prompt in items:
        out = proj / filename
        if out.exists() and not args.force:
            print(f"[skip] đã có {filename}")
            skip += 1
            continue
        try:
            raw = gen(prompt, model, api_key)
            w, h = save_16x9(raw, out)
            print(f"[ok] {filename} ({w}x{h})")
            print(f"SAVED={out}")
            ok += 1
        except Exception as e:  # noqa: BLE001 — báo lỗi từng ảnh, đi tiếp ảnh sau
            print(f"[err] {filename}: {e}", file=sys.stderr)
    print(f"[done] tạo mới {ok}, bỏ qua {skip}")
    return 0 if ok + skip > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
