"""FastAPI application exposing the image similarity API."""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .comparator import BoundingBox, ComparisonError, compare_images

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB per image

app = FastAPI(
    title="Image Similarity Checker",
    description="Compare the same region of two images, robust to lighting changes.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/compare")
async def compare(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    width: int = Form(...),
    height: int = Form(...),
) -> dict:
    # Check the size before reading the file into memory. The upload has already
    # been spooled to a temp file by the server, so `size` is known.
    for label, upload in (("Image 1", image1), ("Image 2", image2)):
        if (upload.size or 0) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"{label} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )

    data1 = await image1.read()
    data2 = await image2.read()

    box = BoundingBox(x=x, y=y, width=width, height=height)
    try:
        # CPU-heavy work runs in a worker thread so it does not block other requests.
        return await run_in_threadpool(compare_images, data1, data2, box)
    except ComparisonError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
