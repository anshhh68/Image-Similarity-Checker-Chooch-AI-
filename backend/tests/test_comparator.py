"""Regression tests for the comparison logic. Run from backend/: `pytest`."""

import base64
import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from app.comparator import BoundingBox, ComparisonError, compare_images
from generate_samples import (
    add_shadow,
    adjust_brightness,
    lower_contrast,
    make_base_scene,
    make_different_scene,
)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


BASE = make_base_scene()
FULL_BOX = BoundingBox(0, 0, *BASE.size)


def test_shadowed_copy_is_similar():
    res = compare_images(png_bytes(BASE), png_bytes(add_shadow(BASE)), FULL_BOX)
    assert res["result"] == "SIMILAR"


def test_different_image_is_not_similar():
    res = compare_images(png_bytes(BASE), png_bytes(make_different_scene()), FULL_BOX)
    assert res["result"] == "NON-SIMILAR"


def test_box_outside_image_is_rejected():
    w, h = BASE.size
    with pytest.raises(ComparisonError, match="does not fit"):
        compare_images(png_bytes(BASE), png_bytes(BASE), BoundingBox(w - 10, 0, 50, 50))


def test_huge_image_is_rejected_before_decoding():
    # 8000x8000 black PNG: tiny on disk, 64 megapixels when decoded.
    ok, encoded = cv2.imencode(".png", np.zeros((8000, 8000), dtype=np.uint8))
    assert ok
    with pytest.raises(ComparisonError, match="too large"):
        compare_images(encoded.tobytes(), png_bytes(BASE), BoundingBox(0, 0, 10, 10))


# --------------------------------------------------------------------------- #
# Difference explanation
# --------------------------------------------------------------------------- #
def explain(other: Image.Image) -> dict:
    return compare_images(png_bytes(BASE), png_bytes(other), FULL_BOX)["analysis"]


def labels(analysis: dict) -> set:
    return {o["label"] for o in analysis["observations"] if o["kind"] != "content"}


def test_identical_has_nothing_to_report():
    a = explain(BASE)
    assert a["summary"].startswith("The regions are practically identical")
    assert labels(a) == set()


def test_shadow_is_described_as_uneven_light_not_blur():
    a = explain(add_shadow(BASE))
    assert a["summary"].startswith("Same content, different conditions")
    assert "uneven lighting" in labels(a)
    assert "sharpness" not in labels(a)
    uneven = next(o for o in a["observations"] if o["label"] == "uneven lighting")
    assert "darker toward the bottom-right" in uneven["detail"]


def test_brightness_and_contrast_are_named_without_false_extras():
    assert labels(explain(adjust_brightness(BASE, 40))) == {"brightness"}
    low = labels(explain(lower_contrast(BASE)))
    assert "contrast" in low and "sharpness" not in low and "colour tint" not in low


def test_blur_is_detected():
    assert "sharpness" in labels(explain(BASE.filter(ImageFilter.GaussianBlur(3))))


def test_partial_change_is_located_and_mapped():
    changed = BASE.copy()
    ImageDraw.Draw(changed).ellipse((60, 60, 200, 200), fill=(30, 30, 160))
    a = explain(changed)
    assert a["summary"].startswith("Mostly the same content")
    assert "top-left" in a["summary"]
    assert "uneven lighting" not in labels(a)  # a new object is not a lighting change
    assert a["difference_map"].startswith("data:image/png;base64,")


def test_different_content_skips_lighting_comparisons():
    a = explain(make_different_scene())
    assert a["summary"].startswith("These regions show different content")
    assert labels(a) == set()


def test_summary_never_contradicts_the_content_finding():
    from app.analysis import _summary

    # Colours differ AND 10% of the shapes changed: must not claim the shapes line up.
    text = _summary(False, 0.10, "the top-left", [])
    assert "line up" not in text and "10%" in text
    # Passes the threshold but 45% changed: must not call that "small".
    text = _summary(True, 0.45, "the left", [])
    assert "small" not in text and "45%" in text


def test_large_previews_are_shrunk():
    big = BASE.resize((2000, 1500))
    res = compare_images(png_bytes(big), png_bytes(big), BoundingBox(0, 0, 2000, 1500))
    for key in ("crop1_preview", "crop2_preview"):
        data = res[key].split(",", 1)[1]
        w, h = Image.open(io.BytesIO(base64.b64decode(data))).size
        assert (w, h) == (768, 576)
