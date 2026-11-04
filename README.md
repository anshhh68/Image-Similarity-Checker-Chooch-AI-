# Image Similarity Checker

Upload two images, enter a bounding box (X, Y, Width, Height), and the app crops
that **same region from both images**, compares their **visual content** and tells
you whether they are **SIMILAR** or **NON-SIMILAR**, with a similarity score.

The comparison is deliberately *not* pixel-by-pixel: it tolerates differences in
brightness, lighting, shadows, reflections and contrast, so the same scene shot
under different light still matches.

---

## Tech stack

| Layer    | Tech |
|----------|------|
| Backend  | Python 3.11, FastAPI, OpenCV (`opencv-python-headless`), scikit-image, NumPy, Pillow |
| Frontend | React 18 + Vite, plain JavaScript, plain CSS (no UI library) |
| Serving  | Uvicorn (API), Nginx (static frontend) |
| Infra    | Docker (one image per service) + Docker Compose |

```
backend/
  app/main.py          FastAPI app, CORS, routes
  app/comparator.py    all image comparison logic
  app/analysis.py      plain-language explanation of the differences
  requirements.txt
  Dockerfile
frontend/
  src/App.jsx  src/App.css  src/main.jsx
  index.html  package.json  vite.config.js  nginx.conf
  Dockerfile           multi-stage: node build -> nginx serve
samples/
  generate_samples.py  generates lighting-variant test pairs
docker-compose.yml
```

---

## Run with Docker (recommended)

```bash
docker compose up --build
```

| What                | URL |
|---------------------|-----|
| Web app             | http://localhost:3000 |
| API                 | http://localhost:8000 |
| Interactive API docs | http://localhost:8000/docs |
| Health check        | http://localhost:8000/health |

The frontend calls the API from the browser, so the API URL is baked into the
frontend bundle at build time (`VITE_API_URL`, default `http://localhost:8000`).
To point it elsewhere, change the `VITE_API_URL` build arg in `docker-compose.yml`
and rebuild.

---

## Run locally without Docker

### Backend

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Run the tests (from `backend/`):

```bash
pip install -r requirements-dev.txt
pytest
```

### Frontend

```bash
cd frontend
npm install
npm run dev            # http://localhost:3000 (stop the Docker frontend first, it uses the same port)
```

To use a different API address, create `frontend/.env.local`:

```
VITE_API_URL=http://localhost:8000
```

### Sample images

```bash
# from the repo root, with the backend requirements installed
python samples/generate_samples.py            # writes samples/output/*.png
python samples/generate_samples.py --check    # also runs the comparator on every pair
python samples/generate_samples.py --base photo.jpg   # use your own base image
```

This produces `base.png` plus six variants: `a_identical`, `b_brightened` (+40),
`c_darkened` (-40), `d_low_contrast`, `e_shadow` (diagonal shadow gradient) and
`f_different` (a different scene). Upload `base.png` as Image A and any variant as
Image B. Cases a–e should be **SIMILAR** and f should be **NON-SIMILAR**.
`--check` prints a table and exits non-zero if any pair does not match the expected result.

---

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /compare`  (multipart/form-data)

| Field    | Type | Description |
|----------|------|-------------|
| `image1` | file | first image |
| `image2` | file | second image |
| `x`      | int  | left edge of the box, in pixels |
| `y`      | int  | top edge of the box, in pixels |
| `width`  | int  | box width, > 0 |
| `height` | int  | box height, > 0 |

The box must fit inside **both** images (`x + width <= image width`,
`y + height <= image height`).

**Example**

```bash
curl -X POST http://localhost:8000/compare \
  -F "image1=@samples/output/base.png" \
  -F "image2=@samples/output/e_shadow.png" \
  -F x=50 -F y=50 -F width=300 -F height=200
```

**200 response**

```json
{
  "result": "SIMILAR",
  "score": 0.9693,
  "threshold": 0.6,
  "details": {
    "structural_similarity": 0.9735,
    "color_similarity": 0.9629
  },
  "crop1_preview": "data:image/png;base64,iVBORw0KGgo...",
  "crop2_preview": "data:image/png;base64,iVBORw0KGgo...",
  "analysis": {
    "summary": "Same content, different conditions. The regions show the same thing; they differ only in brightness, uneven lighting and colour intensity, which the comparison is designed to tolerate.",
    "observations": [
      {
        "kind": "lighting",
        "label": "uneven lighting",
        "title": "Lighting is uneven between the two",
        "detail": "Relative to image A, image B is darker toward the bottom-right and brighter toward the top-left, by up to 19 points. This usually means a shadow, a reflection or light coming from a different direction."
      }
    ],
    "changed_area": 0.0,
    "difference_map": "data:image/png;base64,iVBORw0KGgo..."
  }
}
```

`crop1_preview` / `crop2_preview` are the raw selected regions, so you can confirm
the right area was compared. `analysis` explains *how* the two regions differ (see
[Explaining the differences](#explaining-the-differences)). `kind` is one of
`content`, `lighting`, `color` or `detail`.

**400 response** (invalid box or undecodable image)

```json
{ "detail": "The bounding box (x=600, y=0, w=100, h=100) does not fit inside Image 1, which is 640x480 pixels. x + width must be <= 640 and y + height must be <= 480." }
```

---

## How the comparison works

All of this lives in [`backend/app/comparator.py`](backend/app/comparator.py).

1. **Decode** both uploads into OpenCV BGR arrays.
2. **Validate** the bounding box: positive size, fits inside both images. Otherwise HTTP 400.
3. **Crop** the same region from both images.
4. **Resize** both crops to 256×256 (`cv2.INTER_AREA`) so they are directly comparable.
5. **Lighting normalization**: convert BGR → LAB and apply **CLAHE**
   (`clipLimit=2.0`, `tileGridSize=(8,8)`) to the **L channel only**, then merge back.
6. **Denoise** with a light 3×3 Gaussian blur.
7. **Two scores**:
   - **Structural similarity**: SSIM (`skimage.metrics.structural_similarity`) on the
     grayscale versions of the normalized crops, negatives clamped to 0.
   - **Color similarity**: a 32×32-bin 2D histogram over the **a and b** channels of LAB
     (not L), lightly smoothed, normalized and compared with
     `cv2.compareHist(..., HISTCMP_CORREL)`, negatives clamped to 0.
8. **Final score** = `0.6 × structural + 0.4 × color`.
9. **Verdict**: `SIMILAR` if the final score is at least `0.60`, otherwise `NON-SIMILAR`.

### Why this is robust to lighting

- **LAB separates brightness from colour.** L holds lightness, while a (green↔red) and
  b (blue↔yellow) hold chroma. Lighting changes mostly change L, so the pipeline can
  normalize or ignore L without losing colour information.
- **CLAHE equalizes lightness locally.** It works on an 8×8 grid of tiles, so a global
  brightness shift, a low-contrast image *and* a shadow covering only part of the
  region all get stretched to a similar local lightness distribution. The clip limit
  stops it from amplifying noise.
- **SSIM compares structure, not values.** SSIM compares local means, variances and
  covariance in sliding windows. It asks "are the same edges, shapes and textures in
  the same places?", which survives what is left of the lighting difference after CLAHE.
- **The chroma histogram ignores brightness.** Because it uses only a and b, a brighter
  or darker copy of the same scene has nearly the same histogram. It is also
  position-independent, which complements SSIM. Light smoothing of the histogram
  keeps small chroma shifts (desaturation from low contrast, clipped highlights)
  from landing in disjoint bins.
- **The blur absorbs speckle.** Sensor noise and small specular highlights (reflections)
  would otherwise lower SSIM.

Results on the generated samples (full-image box):

| Pair            | Score | SSIM  | Color | Verdict |
|-----------------|-------|-------|-------|---------|
| identical       | 1.000 | 1.000 | 1.000 | SIMILAR |
| brightened +40  | 0.857 | 0.970 | 0.688 | SIMILAR |
| darkened −40    | 0.963 | 0.951 | 0.982 | SIMILAR |
| low contrast    | 0.791 | 0.952 | 0.551 | SIMILAR |
| shadow gradient | 0.881 | 0.922 | 0.819 | SIMILAR |
| different scene | 0.162 | 0.230 | 0.059 | NON-SIMILAR |

---

## Explaining the differences

The score says *whether* the regions match. The "What's different" section says *how*
they differ, in plain sentences with real numbers. The code is in
[`backend/app/analysis.py`](backend/app/analysis.py). It runs locally and needs no AI
service.

Each check measures one thing and reports it only when the difference is big enough
to matter. The thresholds are constants at the top of the file.

| Check | How it is measured | Example sentence |
|-------|--------------------|------------------|
| Content | Where the per-pixel SSIM map (from the lighting-normalized crops) is low | "Part of the content differs (6% of the region) … concentrated in the top-left" |
| Brightness | Average lightness (LAB L, 0–100) | "Image B is noticeably darker: 64 → 49 (−15 points)" |
| Contrast | Spread of lightness | "lower contrast … 44% smaller" |
| Uneven lighting | Fit `L_B ≈ gain·L_A + offset`, then look for areas the fit can't explain; a tilted-plane fit gives the direction | "darker toward the bottom-right and brighter toward the top-left" |
| Colour tint | Shift of the average a/b colour | "Image B looks warmer (more yellow)" |
| Colour intensity | Average chroma (saturation) | "colours are duller … 26% lower" |
| Sharpness | Finest-detail energy ÷ slightly coarser detail energy, which doesn't change with contrast | "Image B is blurrier" |

Rules that keep the sentences honest:
- Areas where the content changed are left out of the lighting analysis, so a new
  object isn't reported as a shadow.
- When the content is different (15% or more of the area changed), the lighting and
  colour comparisons are skipped, because they would compare different objects.
- The sharpness measure doesn't change with contrast, so a shadow or a flat image
  isn't called blurry.

The **difference map** is image B faded to grey, with the changed areas tinted orange
and outlined.

---

## Tuning

The tunable constants are at the top of `backend/app/comparator.py`:

```python
SIMILARITY_THRESHOLD = 0.60   # raise for stricter matching, lower for looser
STRUCTURAL_WEIGHT = 0.6       # importance of shapes/edges/texture
COLOR_WEIGHT = 0.4            # importance of colour distribution
CLAHE_CLIP_LIMIT = 2.0        # higher = stronger contrast normalization (more noise)
HIST_SMOOTHING_KERNEL = (5, 5)
```

- **Getting false SIMILARs?** Raise `SIMILARITY_THRESHOLD` (for example to 0.70), or move
  weight toward `STRUCTURAL_WEIGHT`.
- **Lighting variants coming out as NON-SIMILAR?** Lower the threshold slightly, or
  raise `STRUCTURAL_WEIGHT`. SSIM after CLAHE is usually the more lighting-stable score.
- Keep `STRUCTURAL_WEIGHT + COLOR_WEIGHT = 1` so the score stays in 0–1.
- A good way to tune is to collect a few real pairs you consider similar and not
  similar, look at their scores in the UI, and put the threshold in the gap between them.

---

## Limitations and possible improvements

- **Alignment-sensitive.** The same box is cropped from both images, so if the object
  moves, scales or rotates between shots, SSIM drops. **ORB feature matching**
  (keypoints + descriptors + RANSAC homography) could align the crops first, or serve
  as a geometry-invariant third score.
- **Low-level features only.** SSIM and histograms measure pixels and colours, not
  meaning. Two different red cars may score low, and two blank walls may score high.
  **CNN embeddings** (for example ResNet, EfficientNet, CLIP or DINOv2 features compared
  with cosine similarity) would capture semantic similarity and are more robust to
  viewpoint and lighting.
- **Strong coloured lighting** (for example sodium-orange street light) shifts a/b, which
  lowers the colour score. A grey-world or other white-balance step before the
  comparison would help.
- **Very small boxes** (a few pixels) are upscaled to 256×256, which makes the scores
  less meaningful. A minimum box size could be enforced.
- **A single fixed threshold** may not suit every domain. It could be calibrated on
  labelled pairs, or replaced with a small classifier trained on the sub-scores.
