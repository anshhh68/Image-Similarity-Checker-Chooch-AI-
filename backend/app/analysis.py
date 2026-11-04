"""
Plain-language explanation of *how* two crops differ.

The similarity score says whether two regions match. This module says why:
it measures brightness, contrast, uneven lighting, colour, sharpness and
structural (content) changes between the two crops, turns each measurement
that is big enough to matter into a short sentence, and renders a map of
where the content differs.

All measurements are done on the crops resized to the same size, so the
numbers are comparable whatever the size of the selected region.
"""

from __future__ import annotations

import cv2
import numpy as np

SIZE = (256, 256)

# When is a difference worth mentioning? Lightness is on a 0-100 scale.
BRIGHTNESS_MIN_POINTS = 4.0       # change in average lightness
CONTRAST_MIN_RATIO = 1.2          # change in lightness spread, either way
UNEVEN_LIGHT_MIN_POINTS = 6.0     # local lighting change not explained globally
TINT_MIN = 7.0                    # shift of average colour (LAB a/b units)
SATURATION_MIN_RATIO = 1.3        # change in colour intensity, either way
SHARPNESS_MIN_RATIO = 2.0         # change in fine-detail energy, either way

CHANGED_DISSIMILARITY = 0.5       # local (1 - SSIM) above this = "content changed"
CHANGED_AREA_MIN = 0.02           # below this share of the region: "structure matches"
DIFFERENT_CONTENT_AREA = 0.15     # above this: the content itself differs

GRID_NAMES = (
    ("top-left", "top", "top-right"),
    ("left", "center", "right"),
    ("bottom-left", "bottom", "bottom-right"),
)

HIGHLIGHT_BGR = np.array([31, 91, 255], dtype=np.float32)  # the UI's orange, #ff5b1f
OUTLINE_BGR = (18, 18, 18)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _grid_means(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Mean of `values` in each cell of a 3x3 grid. Cells with too few valid pixels are NaN."""
    if valid is None:
        valid = np.ones(values.shape, dtype=bool)
    rows = np.array_split(np.arange(values.shape[0]), 3)
    cols = np.array_split(np.arange(values.shape[1]), 3)
    out = np.full((3, 3), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = values[r[0] : r[-1] + 1, c[0] : c[-1] + 1]
            m = valid[r[0] : r[-1] + 1, c[0] : c[-1] + 1]
            if m.sum() >= 0.05 * m.size:
                out[i, j] = v[m].mean()
    return out


def _where(weights: np.ndarray) -> str:
    """Name the part of the region where `weights` (3x3, >= 0) are concentrated."""
    weights = np.nan_to_num(weights, nan=0.0).clip(min=0)
    total = weights.sum()
    if total <= 0:
        return "the region"
    share = weights / total
    if (share >= 0.07).sum() >= 7:
        return "the whole region"
    order = np.argsort(share, axis=None)[::-1]
    names = [GRID_NAMES[k // 3][k % 3] for k in order[:3] if share.flat[k] >= 0.18]
    if not names:
        names = [GRID_NAMES[order[0] // 3][order[0] % 3]]
    if len(names) == 1:
        return f"the {names[0]}"
    return "the " + ", ".join(names[:-1]) + f" and {names[-1]}"


def _observation(kind: str, label: str, title: str, detail: str) -> dict:
    """kind groups the observation in the UI; label is the short name used in the summary."""
    return {"kind": kind, "label": label, "title": title, "detail": detail}


# --------------------------------------------------------------------------- #
# Individual measurements. Each returns a list of 0 or 1 observations.
# `l1`/`l2` are lightness (0-100); `lab` arrays are float LAB with a/b centred on 0.
# --------------------------------------------------------------------------- #
def _brightness(l1: np.ndarray, l2: np.ndarray) -> list[dict]:
    m1, m2 = l1.mean(), l2.mean()
    d = m2 - m1
    if abs(d) < BRIGHTNESS_MIN_POINTS:
        return []
    word = "slightly" if abs(d) < 10 else "noticeably" if abs(d) < 25 else "much"
    direction = "brighter" if d > 0 else "darker"
    return [
        _observation(
            "lighting",
            "brightness",
            f"Image B is {word} {direction}",
            f"Average lightness goes from {m1:.0f} to {m2:.0f} on a 0–100 scale "
            f"({d:+.0f} points). The comparison evens this out before scoring.",
        )
    ]


def _contrast(l1: np.ndarray, l2: np.ndarray) -> list[dict]:
    s1, s2 = l1.std(), l2.std()
    if s1 < 1 or s2 < 1:
        return []
    r = s2 / s1
    if 1 / CONTRAST_MIN_RATIO < r < CONTRAST_MIN_RATIO:
        return []
    lower = r < 1
    return [
        _observation(
            "lighting",
            "contrast",
            f"Image B has {'lower' if lower else 'higher'} contrast",
            f"The gap between its light and dark areas is {abs(r - 1) * 100:.0f}% "
            f"{'smaller' if lower else 'larger'} (lightness spread {s1:.0f} → {s2:.0f}), "
            f"so it looks {'flatter and more washed out' if lower else 'punchier'}.",
        )
    ]


def _uneven_lighting(l1: np.ndarray, l2: np.ndarray, changed: np.ndarray) -> list[dict]:
    """
    Detect a shadow or a change of light direction.

    A uniform brightness or contrast change is a straight-line relation
    l2 ≈ gain * l1 + offset over the whole crop. We fit that line, then look
    at what is left over in each part of the region: if one area is clearly
    darker (or brighter) than the line predicts, the light changed locally.
    Near-black and near-white pixels are ignored because clipping distorts them,
    and so are areas whose content changed (`changed`), since a different
    object is not a lighting change.
    """
    valid = (l1 > 4) & (l1 < 96) & (l2 > 4) & (l2 < 96) & ~changed
    if valid.sum() < 500:
        return []
    gain, offset = np.polyfit(l1[valid], l2[valid], 1)
    residual = l2 - (gain * l1 + offset)
    cells = _grid_means(residual, valid)
    if np.isnan(cells).all():
        return []
    spread = np.nanmax(cells) - np.nanmin(cells)
    if spread < UNEVEN_LIGHT_MIN_POINTS:
        return []
    darker, brighter = _light_direction(residual, valid, cells)
    return [
        _observation(
            "lighting",
            "uneven lighting",
            "Lighting is uneven between the two",
            f"Relative to image A, image B is darker toward the {darker} and brighter toward "
            f"the {brighter}, by up to {spread:.0f} points. This usually means a shadow, "
            "a reflection or light coming from a different direction.",
        )
    ]


def _light_direction(residual: np.ndarray, valid: np.ndarray, cells: np.ndarray) -> tuple[str, str]:
    """
    Name the darker and brighter side of a lighting change.

    A shadow edge or light falloff is usually a smooth slope, so we fit a tilted
    plane (residual ≈ a·x + b·y + c) and read its direction. If the plane does
    not explain the change (e.g. a round shadow in the middle), we fall back to
    the darkest and brightest 3x3 cell.
    """
    h, w = residual.shape
    ys, xs = np.nonzero(valid)
    design = np.column_stack([xs / (w - 1), ys / (h - 1), np.ones(xs.size)])
    (a, b, _), *_ = np.linalg.lstsq(design, residual[valid], rcond=None)
    plane_range = abs(a) + abs(b)
    cell_spread = np.nanmax(cells) - np.nanmin(cells)

    if plane_range >= 0.6 * cell_spread:
        length = np.hypot(a, b)
        dx, dy = -a / length, -b / length          # unit vector pointing to the darker side

        def side(ux, uy):
            col = 0 if ux < -0.38 else 2 if ux > 0.38 else 1   # 0.38 ≈ cos(67.5°)
            row = 0 if uy < -0.38 else 2 if uy > 0.38 else 1
            return GRID_NAMES[row][col]

        return side(dx, dy), side(-dx, -dy)

    dark = np.unravel_index(np.nanargmin(cells), cells.shape)
    bright = np.unravel_index(np.nanargmax(cells), cells.shape)
    return GRID_NAMES[dark[0]][dark[1]], GRID_NAMES[bright[0]][bright[1]]


def _tint(lab1: np.ndarray, lab2: np.ndarray) -> list[dict]:
    da = lab2[..., 1].mean() - lab1[..., 1].mean()   # + = redder,  - = greener
    db = lab2[..., 2].mean() - lab1[..., 2].mean()   # + = yellower, - = bluer
    parts = []
    if db >= TINT_MIN:
        parts.append("warmer (more yellow)")
    elif db <= -TINT_MIN:
        parts.append("cooler (more blue)")
    if da >= TINT_MIN:
        parts.append("more red or magenta")
    elif da <= -TINT_MIN:
        parts.append("more green")
    if not parts:
        return []
    return [
        _observation(
            "color",
            "colour tint",
            f"Image B looks {' and '.join(parts)}",
            f"The average colour shifts by {da:+.1f} on the green–red axis and "
            f"{db:+.1f} on the blue–yellow axis. This points to a different light "
            "source or white balance, or to differently coloured objects.",
        )
    ]


def _saturation(lab1: np.ndarray, lab2: np.ndarray) -> list[dict]:
    c1 = np.hypot(lab1[..., 1], lab1[..., 2]).mean()
    c2 = np.hypot(lab2[..., 1], lab2[..., 2]).mean()
    if c1 < 3:  # image A is (almost) grey: a ratio would be meaningless
        return []
    r = c2 / c1
    if 1 / SATURATION_MIN_RATIO < r < SATURATION_MIN_RATIO:
        return []
    less = r < 1
    return [
        _observation(
            "color",
            "colour intensity",
            f"Image B's colours are {'duller' if less else 'more vivid'}",
            f"Colour intensity (saturation) is {abs(r - 1) * 100:.0f}% "
            f"{'lower' if less else 'higher'} than in image A, at the same hues.",
        )
    ]


def _sharpness(gray1: np.ndarray, gray2: np.ndarray) -> list[dict]:
    """
    Sharpness = finest detail energy / slightly coarser detail energy.

    Lowering contrast or adding a shadow weakens *all* edges by the same
    factor, so the ratio stays the same. Blur removes the finest detail first,
    so the ratio drops. That keeps lighting changes from being called blur.
    """
    def sharpness(g):
        fine = cv2.Laplacian(g, cv2.CV_64F).var()
        coarse = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 2), cv2.CV_64F).var()
        return fine / coarse if coarse > 1e-6 else 0.0

    v1, v2 = sharpness(gray1), sharpness(gray2)
    if v1 <= 0 or v2 <= 0:  # (almost) flat crop: nothing to compare
        return []
    r = v2 / v1
    if 1 / SHARPNESS_MIN_RATIO < r < SHARPNESS_MIN_RATIO:
        return []
    blurrier = r < 1
    change = (1 - r) * 100 if blurrier else (r - 1) * 100
    return [
        _observation(
            "detail",
            "sharpness",
            f"Image B is {'blurrier' if blurrier else 'sharper'}",
            f"Fine detail and edge crispness are about {change:.0f}% "
            f"{'lower' if blurrier else 'higher'}, as caused by focus, motion, "
            "noise or compression.",
        )
    ]


def _content(ssim_map: np.ndarray, ssim: float) -> tuple[list[dict], float, str, np.ndarray, np.ndarray]:
    """Where does the structure (edges, shapes, texture) differ?"""
    dissim = np.clip(1.0 - ssim_map, 0.0, 1.0).astype(np.float32)
    dissim = cv2.GaussianBlur(dissim, (0, 0), 4)          # merge speckles into areas
    mask = dissim > CHANGED_DISSIMILARITY
    area = float(mask.mean())
    where = _where(_grid_means(mask.astype(np.float32)))

    if area < CHANGED_AREA_MIN:
        obs = _observation(
            "content",
            "structure",
            "Shapes and edges match",
            "Once lighting is evened out, the edges, shapes and textures line up "
            f"across the region (structural similarity {ssim * 100:.0f}%).",
        )
    elif area >= 0.6:
        obs = _observation(
            "content",
            "structure",
            "The content is different almost everywhere",
            f"{area * 100:.0f}% of the region has different shapes, edges or texture. "
            "These are not the same object or scene.",
        )
    else:
        obs = _observation(
            "content",
            "structure",
            f"Part of the content differs ({area * 100:.0f}% of the region)",
            f"The changed shapes, edges or texture are concentrated in {where}. "
            "The difference map highlights exactly where.",
        )
    return [obs], area, where, dissim, mask


def _summary(similar: bool, area: float, where: str, conditions: list[str]) -> str:
    if not similar:
        if area >= DIFFERENT_CONTENT_AREA:
            return (
                f"These regions show different content: {area * 100:.0f}% of the area has "
                "different shapes, edges or texture, so lighting alone cannot explain it."
            )
        if area >= CHANGED_AREA_MIN:
            return (
                f"The colours differ, and part of the content too: {area * 100:.0f}% of the area "
                f"has different shapes, edges or texture, around {where}."
            )
        return (
            "The layout matches but the colours do not: shapes and edges line up, yet the "
            "mix of colours is different enough to fall below the threshold."
        )
    if area >= DIFFERENT_CONTENT_AREA:
        return (
            f"Partly the same content. {area * 100:.0f}% of the region differs in structure, "
            f"around {where}, but the rest matches closely enough to pass."
        )
    if area >= CHANGED_AREA_MIN:
        return (
            f"Mostly the same content. A small part ({area * 100:.0f}%) differs in structure, "
            f"around {where}; the rest matches."
        )
    if conditions:
        listed = conditions[0] if len(conditions) == 1 else ", ".join(conditions[:-1]) + f" and {conditions[-1]}"
        return (
            f"Same content, different conditions. The regions show the same thing; they "
            f"differ only in {listed}, which the comparison is designed to tolerate."
        )
    return "The regions are practically identical. No meaningful difference in content, lighting or colour was found."


def _difference_map(crop2: np.ndarray, dissim: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Image B, faded, with changed areas tinted orange and outlined."""
    h, w = crop2.shape[:2]
    heat = cv2.resize(dissim, (w, h), interpolation=cv2.INTER_LINEAR)
    outline_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    base = 0.45 * gray + 0.55 * 235.0                      # light, faded backdrop
    base = np.repeat(base[:, :, None], 3, axis=2)
    # At most half-strength orange, so image B stays readable even when most of it changed.
    alpha = (np.clip((heat - 0.15) / 0.5, 0, 1) * 0.5)[:, :, None]
    out = (base * (1 - alpha) + HIGHLIGHT_BGR * alpha).astype(np.uint8)

    contours, _ = cv2.findContours(outline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, OUTLINE_BGR, max(1, round(max(w, h) / 200)))
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def describe_differences(
    crop1: np.ndarray,
    crop2: np.ndarray,
    ssim_map: np.ndarray,
    ssim: float,
    similar: bool,
) -> tuple[dict, np.ndarray]:
    """
    Explain how crop2 differs from crop1. `ssim_map` is the per-pixel SSIM
    of the lighting-normalised crops.

    Returns (analysis, difference_map) where analysis is
    {"summary": str, "observations": [{kind, label, title, detail}], "changed_area": float}
    and difference_map is a BGR image the size of crop2.
    """
    raw1 = cv2.resize(crop1, SIZE, interpolation=cv2.INTER_AREA)
    raw2 = cv2.resize(crop2, SIZE, interpolation=cv2.INTER_AREA)

    lab1 = cv2.cvtColor(raw1, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(raw2, cv2.COLOR_BGR2LAB).astype(np.float32)
    for lab in (lab1, lab2):
        lab[..., 0] *= 100.0 / 255.0   # OpenCV stores L as 0-255; use the usual 0-100
        lab[..., 1:] -= 128.0          # centre a/b on 0 (0 = neutral grey)
    l1, l2 = lab1[..., 0], lab2[..., 0]

    content_obs, area, where, dissim, mask = _content(ssim_map, ssim)

    if area >= DIFFERENT_CONTENT_AREA:
        # Different objects: comparing their lighting or colour would mislead.
        conditions = []
    else:
        conditions = (
            _brightness(l1, l2)
            + _contrast(l1, l2)
            + _uneven_lighting(l1, l2, mask)
            + _tint(lab1, lab2)
            + _saturation(lab1, lab2)
            + _sharpness(
                cv2.cvtColor(raw1, cv2.COLOR_BGR2GRAY).astype(np.float64),
                cv2.cvtColor(raw2, cv2.COLOR_BGR2GRAY).astype(np.float64),
            )
        )

    names = list(dict.fromkeys(obs["label"] for obs in conditions))  # unique, in order

    analysis = {
        "summary": _summary(similar, area, where, names),
        "observations": content_obs + conditions,
        "changed_area": round(area, 4),
    }
    return analysis, _difference_map(crop2, dissim, mask)
