"""Draw the app icon (two speech bubbles) as a macOS .iconset folder. Needs Pillow."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw


def draw(size: int) -> Image.Image:
    s = size / 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Rounded-square background, violet -> blue vertical gradient.
    grad = Image.new("RGBA", (size, size))
    gd = ImageDraw.Draw(grad)
    for y in range(size):
        t = y / max(size - 1, 1)
        gd.line([(0, y), (size, y)], fill=(int(124 - 60 * t), int(58 + 40 * t), int(237 - 20 * t), 255))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([int(80 * s), int(80 * s), int(944 * s), int(944 * s)], int(190 * s), fill=255)
    img.paste(grad, (0, 0), mask)

    # Back bubble (translucent) and front bubble (white), each with a tail.
    d.rounded_rectangle([int(330 * s), int(230 * s), int(820 * s), int(560 * s)], int(90 * s), fill=(255, 255, 255, 140))
    d.polygon([(int(690 * s), int(550 * s)), (int(770 * s), int(650 * s)), (int(620 * s), int(555 * s))], fill=(255, 255, 255, 140))
    d.rounded_rectangle([int(200 * s), int(420 * s), int(690 * s), int(750 * s)], int(90 * s), fill=(255, 255, 255, 255))
    d.polygon([(int(300 * s), int(740 * s)), (int(250 * s), int(850 * s)), (int(390 * s), int(745 * s))], fill=(255, 255, 255, 255))
    # "Sound bars" inside the front bubble.
    bars = [(280, 110), (360, 190), (440, 140), (520, 210), (600, 120)]
    for x, h in bars:
        cy = 585
        d.rounded_rectangle(
            [int((x - 22) * s), int((cy - h / 2) * s), int((x + 22) * s), int((cy + h / 2) * s)],
            int(22 * s),
            fill=(92, 72, 230, 255),
        )
    return img


def main(out_dir: str) -> None:
    iconset = Path(out_dir)
    iconset.mkdir(parents=True, exist_ok=True)
    for base in (16, 32, 128, 256, 512):
        draw(base).save(iconset / f"icon_{base}x{base}.png")
        draw(base * 2).save(iconset / f"icon_{base}x{base}@2x.png")


if __name__ == "__main__":
    main(sys.argv[1])
