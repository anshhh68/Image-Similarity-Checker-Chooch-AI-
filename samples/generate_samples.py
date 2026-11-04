"""
Generate test image pairs for the Image Similarity Checker.

From one base image this script produces:
    (a) identical         - exact copy
    (b) brightened        - +40 on every channel
    (c) darkened          - -40 on every channel
    (d) low_contrast      - contrast reduced to 55%
    (e) shadow            - a dark diagonal shadow gradient overlaid
    (f) different         - a clearly different image (different scene)

Cases (a)-(e) should come out as SIMILAR, (f) as NON-SIMILAR.

Usage:
    python samples/generate_samples.py                 # synthetic base image
    python samples/generate_samples.py --base my.jpg   # use your own base image
    python samples/generate_samples.py --check         # also run the comparator on every pair

Output goes to samples/output/ (or --out DIR).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = (640, 480)  # width, height of the synthetic images


# --------------------------------------------------------------------------- #
# Synthetic scenes
# --------------------------------------------------------------------------- #
def _gradient(size, top, bottom) -> Image.Image:
    """Vertical colour gradient."""
    w, h = size
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    top_arr = np.array(top, dtype=np.float32)[None, None, :]
    bottom_arr = np.array(bottom, dtype=np.float32)[None, None, :]
    arr = top_arr * (1 - t) + bottom_arr * t
    arr = np.repeat(arr, w, axis=1)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def make_base_scene() -> Image.Image:
    """A textured scene with shapes in several colours: sky, hills, sun, house."""
    rng = np.random.default_rng(7)
    img = _gradient(SIZE, (120, 180, 235), (220, 235, 250))
    draw = ImageDraw.Draw(img)
    w, h = SIZE

    # sun
    draw.ellipse((470, 50, 570, 150), fill=(255, 200, 60))
    # hills
    draw.polygon([(0, 330), (160, 230), (330, 320), (480, 220), (640, 300), (640, 480), (0, 480)],
                 fill=(70, 150, 80))
    draw.polygon([(0, 400), (200, 320), (420, 390), (640, 340), (640, 480), (0, 480)],
                 fill=(45, 115, 60))
    # house
    draw.rectangle((150, 270, 300, 390), fill=(200, 80, 60))
    draw.polygon([(135, 275), (225, 200), (315, 275)], fill=(110, 50, 40))
    draw.rectangle((205, 320, 245, 390), fill=(90, 55, 35))
    draw.rectangle((165, 295, 195, 325), fill=(240, 240, 200))
    draw.rectangle((255, 295, 285, 325), fill=(240, 240, 200))
    # tree
    draw.rectangle((420, 300, 440, 380), fill=(100, 70, 40))
    draw.ellipse((380, 220, 480, 320), fill=(30, 120, 50))
    # path / stripes for structure
    for i in range(6):
        draw.line((0, 420 + i * 10, w, 405 + i * 12), fill=(190, 170, 120), width=3)

    # fine texture so SSIM has real structure to work with
    arr = np.asarray(img).astype(np.int16)
    arr += rng.integers(-8, 9, size=arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def make_different_scene() -> Image.Image:
    """A clearly different scene: checkerboard, purple tones, random circles."""
    rng = np.random.default_rng(42)
    img = _gradient(SIZE, (60, 20, 90), (160, 60, 140))
    draw = ImageDraw.Draw(img)
    w, h = SIZE

    cell = 40
    for yy in range(0, h, cell):
        for xx in range(0, w, cell):
            if (xx // cell + yy // cell) % 2 == 0:
                draw.rectangle((xx, yy, xx + cell - 1, yy + cell - 1), fill=(240, 220, 90))
    for _ in range(25):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = int(rng.integers(10, 50))
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    return img.filter(ImageFilter.GaussianBlur(0.6))


# --------------------------------------------------------------------------- #
# Lighting variants
# --------------------------------------------------------------------------- #
def adjust_brightness(img: Image.Image, delta: int) -> Image.Image:
    arr = np.asarray(img).astype(np.int16) + delta
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def lower_contrast(img: Image.Image, factor: float = 0.55) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    mean = arr.mean()
    arr = (arr - mean) * factor + mean
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def add_shadow(img: Image.Image, strength: float = 0.6) -> Image.Image:
    """Multiply by a diagonal gradient: full brightness top-left, dark bottom-right."""
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    t = (xx / (w - 1) + yy / (h - 1)) / 2.0           # 0 .. 1 diagonally
    mask = 1.0 - strength * t                          # 1.0 .. 1-strength
    arr *= mask[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_pairs(base: Image.Image) -> dict[str, Image.Image]:
    different = make_different_scene().resize(base.size)
    return {
        "a_identical": base.copy(),
        "b_brightened": adjust_brightness(base, +40),
        "c_darkened": adjust_brightness(base, -40),
        "d_low_contrast": lower_contrast(base),
        "e_shadow": add_shadow(base),
        "f_different": different,
    }


def run_check(out_dir: Path, base_path: Path, pairs: dict[str, Image.Image]) -> int:
    """Run the backend comparator on every pair and print the results."""
    backend_dir = Path(__file__).resolve().parent.parent / "backend"
    sys.path.insert(0, str(backend_dir))
    from app.comparator import BoundingBox, compare_images  # noqa: E402

    base_bytes = base_path.read_bytes()
    w, h = Image.open(base_path).size
    boxes = {
        "full image": BoundingBox(0, 0, w, h),
        "center region": BoundingBox(w // 4, h // 4, w // 2, h // 2),
    }

    failures = 0
    print()
    for box_name, box in boxes.items():
        print(f"Bounding box: {box_name} ({box.x}, {box.y}, {box.width}, {box.height})")
        print(f"  {'pair':<16}{'expected':<14}{'result':<14}{'score':>7}{'ssim':>8}{'color':>8}")
        for name in pairs:
            other = (out_dir / f"{name}.png").read_bytes()
            res = compare_images(base_bytes, other, box)
            expected = "NON-SIMILAR" if name.startswith("f_") else "SIMILAR"
            ok = res["result"] == expected
            failures += 0 if ok else 1
            d = res["details"]
            print(
                f"  {name:<16}{expected:<14}{res['result']:<14}{res['score']:>7.3f}"
                f"{d['structural_similarity']:>8.3f}{d['color_similarity']:>8.3f}"
                f"  {'ok' if ok else 'MISMATCH'}"
            )
        print()
    print("All checks passed." if failures == 0 else f"{failures} mismatch(es).")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, help="Optional base image (default: synthetic scene).")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output",
                        help="Output directory (default: samples/output).")
    parser.add_argument("--check", action="store_true",
                        help="Run the backend comparator on each pair (needs backend requirements).")
    args = parser.parse_args()

    base = Image.open(args.base).convert("RGB") if args.base else make_base_scene()
    args.out.mkdir(parents=True, exist_ok=True)

    base_path = args.out / "base.png"
    base.save(base_path)
    pairs = build_pairs(base)
    for name, img in pairs.items():
        img.save(args.out / f"{name}.png")

    print(f"Wrote base.png and {len(pairs)} variants to {args.out}")
    print("Upload base.png as Image A and any variant as Image B.")

    if args.check:
        return 1 if run_check(args.out, base_path, pairs) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
