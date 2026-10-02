import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _resolve_font(prefer_cjk: bool) -> str | None:
    """Tìm font dùng được trên Windows/macOS/Linux; None nếu không thấy."""
    if prefer_cjk:
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",  # Windows
            "/System/Library/Fonts/Hiragino Sans GB.ttc",  # macOS
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS fallback
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS (hỗ trợ tiếng Việt)
            "/System/Library/Fonts/Helvetica.ttc",  # macOS fallback
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def _load_font(path: str | None, size: int):
    if path:
        return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow cũ: load_default() không nhận size
        return ImageFont.load_default()


def main(image_path: str, annotation_path: str, output_path: str) -> None:
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Arial supports Vietnamese; retain the CJK font for Chinese annotations.
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    labels = " ".join(element["label"] for element in data["elements"])
    prefer_cjk = any("\u4e00" <= c <= "\u9fff" for c in labels)
    font = _load_font(_resolve_font(prefer_cjk), 28)
    small_font = _load_font(_resolve_font(prefer_cjk), 18)
    colors = [(38, 103, 255, 225), (255, 105, 92, 225), (41, 167, 102, 225), (181, 100, 255, 225)]

    for index, element in enumerate(data["elements"], start=1):
        region = element["region"]
        x, y = region["x"], region["y"]
        right, bottom = x + region["width"], y + region["height"]
        color = colors[(index - 1) % len(colors)]
        fill = (*color[:3], 24)
        draw.rounded_rectangle((x, y, right, bottom), radius=12, outline=color, width=4, fill=fill)
        draw.ellipse((x + 8, y + 8, x + 44, y + 44), fill=color)
        draw.text((x + 19, y + 8), str(index), anchor="ma", font=small_font, fill="white")
        label = f"{index}. {element['label']}  {element['reveal']['direction']}"
        draw.rounded_rectangle((x + 52, y + 8, min(right - 8, x + 52 + len(label) * 19), y + 46), radius=6, fill=(255, 255, 255, 225))
        draw.text((x + 60, y + 12), label, font=small_font, fill=color)
        start = tuple(element["handPath"]["start"])
        end = tuple(element["handPath"]["end"])
        draw.line((start, end), fill=color, width=4)
        draw.polygon((end, (end[0] - 13, end[1] - 7), (end[0] - 13, end[1] + 7)), fill=color)

    result = Image.alpha_composite(image, overlay).convert("RGB")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, quality=95)


if __name__ == "__main__":
    main(*sys.argv[1:4])
