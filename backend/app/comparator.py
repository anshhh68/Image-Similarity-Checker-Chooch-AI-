"""
Image comparison logic.

The goal is to decide whether the same rectangular region of two images shows
the same *visual content*, while being tolerant to lighting changes such as
brightness shifts, shadows, reflections and contrast differences.

Pipeline (see `compare_images`):
    decode -> validate box -> crop -> resize -> CLAHE on L channel (LAB)
    -> light blur -> SSIM (structure) + a/b chroma histogram (colour)
    -> weighted score -> verdict
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .analysis import describe_differences

# --------------------------------------------------------------------------- #
# Tunable constants
# --------------------------------------------------------------------------- #
SIMILARITY_THRESHOLD = 0.60      # final_score >= this  ->  "SIMILAR"
STRUCTURAL_WEIGHT = 0.6          # weight of the SSIM score
COLOR_WEIGHT = 0.4               # weight of the chroma histogram score

RESIZE_DIM = (256, 256)          # both crops are resized to this before comparing
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
BLUR_KERNEL = (3, 3)
HIST_BINS = [32, 32]             # bins for the a and b channels
HIST_RANGES = [0, 256, 0, 256]   # OpenCV 8-bit LAB stores a/b in 0..255
HIST_SMOOTHING_KERNEL = (5, 5)   # Gaussian smoothing of the 2D a/b histogram

# Reject images larger than this many pixels *before* decoding them. A tiny
# compressed file (e.g. a 1 MB all-black PNG) can decode to gigabytes of RAM.
MAX_IMAGE_PIXELS = 50_000_000

PREVIEW_MAX_SIDE = 768           # longest side of the preview images sent back


class ComparisonError(ValueError):
    """Raised for invalid user input (bad image, bad bounding box)."""


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


# --------------------------------------------------------------------------- #
# Step 1: decoding
# --------------------------------------------------------------------------- #
def decode_image(data: bytes, label: str) -> np.ndarray:
    """Decode raw upload bytes into an OpenCV BGR array."""
    if not data:
        raise ComparisonError(f"{label} is empty.")

    invalid = ComparisonError(
        f"{label} could not be decoded. Please upload a valid image (PNG, JPEG, BMP, WEBP...)."
    )
    too_large = ComparisonError(
        f"{label} is too large to process (max {MAX_IMAGE_PIXELS // 1_000_000} megapixels)."
    )

    # Read only the header to get the dimensions; this does not decode pixels.
    try:
        with Image.open(io.BytesIO(data)) as header:
            width, height = header.size
    except Image.DecompressionBombError as exc:
        raise too_large from exc
    except Exception as exc:
        raise invalid from exc
    if width * height > MAX_IMAGE_PIXELS:
        raise too_large

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise invalid
    return image


# --------------------------------------------------------------------------- #
# Step 2: validation
# --------------------------------------------------------------------------- #
def validate_box(box: BoundingBox, image: np.ndarray, label: str) -> None:
    """Make sure the bounding box is well-formed and fits inside the image."""
    img_h, img_w = image.shape[:2]

    if box.width <= 0 or box.height <= 0:
        raise ComparisonError("Width and height must both be greater than 0.")
    if box.x < 0 or box.y < 0:
        raise ComparisonError("X and Y must be 0 or greater.")
    if box.x + box.width > img_w or box.y + box.height > img_h:
        raise ComparisonError(
            f"The bounding box (x={box.x}, y={box.y}, w={box.width}, h={box.height}) "
            f"does not fit inside {label}, which is {img_w}x{img_h} pixels. "
            f"x + width must be <= {img_w} and y + height must be <= {img_h}."
        )


# --------------------------------------------------------------------------- #
# Steps 3-6: crop and normalise
# --------------------------------------------------------------------------- #
def crop(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Cut the bounding box region out of the image."""
    return image[box.y : box.y + box.height, box.x : box.x + box.width]


def normalize_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Resize and lighting-normalise a crop. Returns a BGR image.

    - Resize to a fixed size so both crops are directly comparable.
    - CLAHE (Contrast Limited Adaptive Histogram Equalisation) on the L
      (lightness) channel of LAB equalises brightness/contrast *locally*, so a
      global brightness shift, a darker half (shadow) or reduced contrast all
      map to a similar lightness distribution. The a/b (colour) channels are
      left untouched, so hue information is preserved.
    - A light Gaussian blur removes sensor noise and small specular highlights
      (reflection speckle) that would otherwise hurt SSIM.
    """
    resized = cv2.resize(crop_bgr, RESIZE_DIM, interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_channel = clahe.apply(l_channel)
    normalized = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    return cv2.GaussianBlur(normalized, BLUR_KERNEL, 0)


# --------------------------------------------------------------------------- #
# Step 7: scores
# --------------------------------------------------------------------------- #
def structural_score(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> tuple[float, np.ndarray]:
    """
    SSIM on grayscale. SSIM compares local patterns of luminance, contrast and
    structure, so it measures whether the same edges/shapes/textures are
    present rather than whether pixel values match exactly.

    Returns the overall score and the per-pixel SSIM map (used to show *where*
    the two crops differ).
    """
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    score, ssim_map = structural_similarity(gray1, gray2, data_range=255, full=True)
    return _clamp(score), ssim_map


def color_score(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    """
    Correlation of 2D histograms over the a and b channels of LAB.

    a/b encode chroma (green-red and blue-yellow) independently from
    lightness, so this compares "which colours are present" while ignoring
    how bright they are.
    """
    hists = []
    for img in (img1_bgr, img2_bgr):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [1, 2], None, HIST_BINS, HIST_RANGES)
        # Lightly smooth the histogram so that a small chroma shift (e.g. the
        # slight desaturation caused by lower contrast or clipped highlights)
        # spills into neighbouring bins instead of landing in a disjoint bin.
        # Without this, images with flat colours produce spiky histograms
        # whose correlation collapses on tiny shifts.
        hist = cv2.GaussianBlur(hist, HIST_SMOOTHING_KERNEL, 0)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hists.append(hist)
    score = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)
    return _clamp(score)


def _clamp(value: float) -> float:
    """Clamp to [0, 1] and guard against NaN (e.g. degenerate histograms)."""
    value = float(value)
    if np.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


# --------------------------------------------------------------------------- #
# Step 10: previews
# --------------------------------------------------------------------------- #
def to_png_data_url(image_bgr: np.ndarray) -> str:
    """
    Encode a BGR image as a base64 PNG data URL for the frontend.

    Previews are only for looking at, so large ones are shrunk to at most
    PREVIEW_MAX_SIDE pixels on the longest side; a full-size 12 MP crop would
    otherwise add ~20 MB to the response per image.
    """
    h, w = image_bgr.shape[:2]
    if max(h, w) > PREVIEW_MAX_SIDE:
        scale = PREVIEW_MAX_SIDE / max(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode preview image.")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def compare_images(image1_bytes: bytes, image2_bytes: bytes, box: BoundingBox) -> dict:
    """Run the full comparison pipeline and return the API response payload."""
    # 1. decode
    image1 = decode_image(image1_bytes, "Image 1")
    image2 = decode_image(image2_bytes, "Image 2")

    # 2. validate
    validate_box(box, image1, "Image 1")
    validate_box(box, image2, "Image 2")

    # 3. crop the same region from both images
    crop1 = crop(image1, box)
    crop2 = crop(image2, box)

    # 4-6. resize, lighting normalisation, blur
    norm1 = normalize_crop(crop1)
    norm2 = normalize_crop(crop2)

    # 7. sub-scores
    ssim, ssim_map = structural_score(norm1, norm2)
    color = color_score(norm1, norm2)

    # 8. weighted final score
    final_score = STRUCTURAL_WEIGHT * ssim + COLOR_WEIGHT * color

    # 9. verdict
    result = "SIMILAR" if final_score >= SIMILARITY_THRESHOLD else "NON-SIMILAR"

    # Plain-language explanation of what differs, plus a map of where.
    analysis, difference_map = describe_differences(
        crop1, crop2, ssim_map, ssim, similar=result == "SIMILAR"
    )
    analysis["difference_map"] = to_png_data_url(difference_map)

    # 10. previews of the raw crops (what the user actually selected)
    return {
        "analysis": analysis,
        "result": result,
        "score": round(final_score, 4),
        "threshold": SIMILARITY_THRESHOLD,
        "details": {
            "structural_similarity": round(ssim, 4),
            "color_similarity": round(color, 4),
        },
        "crop1_preview": to_png_data_url(crop1),
        "crop2_preview": to_png_data_url(crop2),
    }
