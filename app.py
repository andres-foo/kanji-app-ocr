"""FastAPI entry point for the local KanjiOCRVibe OCR service."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

from ocr_engine import OCRNotReady, OCREngine

logger = logging.getLogger("kanjiocrvibe.app")

INVALID_IMAGE = {"ok": False, "error": "invalid image"}
OCR_FAILED = {"ok": False, "error": "ocr failed"}
LOADING = {"ok": False, "error": "loading"}


class InvalidImageError(ValueError):
    """The request body is not a decodable JPEG."""


@dataclass(frozen=True)
class DecodedImage:
    width: int
    height: int
    bgr: np.ndarray


def decode_jpeg(raw: bytes) -> DecodedImage:
    """Strictly decode JPEG bytes, apply EXIF orientation, return BGR pixels.

    PaddleOCR expects OpenCV-style BGR arrays.  Nothing is written to disk.
    """

    if not raw or len(raw) < 4 or raw[:2] != b"\xff\xd8":
        raise InvalidImageError("not a JPEG")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "JPEG":
                raise InvalidImageError("not a JPEG")

            source.load()
            oriented = ImageOps.exif_transpose(source)
            rgb_image = oriented.convert("RGB")
            width, height = rgb_image.size
            if width <= 0 or height <= 0:
                raise InvalidImageError("invalid dimensions")

            rgb = np.asarray(rgb_image, dtype=np.uint8)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    except InvalidImageError:
        raise
    except Exception as exc:
        raise InvalidImageError("undecodable JPEG") from exc

    return DecodedImage(width=width, height=height, bgr=bgr)


def _content_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()


def create_app(engine: OCREngine | None = None) -> FastAPI:
    """Build the app.  ``engine`` is injectable for offline tests."""

    ocr_engine = engine if engine is not None else OCREngine()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Starts a daemon thread and returns immediately.
        ocr_engine.start_loading()
        yield

    app = FastAPI(
        title="KanjiOCRVibe OCR",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.engine = ocr_engine

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        state = request.app.state.engine.state()
        # Health is always HTTP 200, including a failed model load.
        return JSONResponse(status_code=200, content=state.as_dict())

    @app.post("/ocr")
    async def ocr(request: Request) -> JSONResponse:
        service: OCREngine = request.app.state.engine
        if not service.state().ready:
            return JSONResponse(status_code=503, content=LOADING)

        if _content_type(request) != "image/jpeg":
            return JSONResponse(status_code=400, content=INVALID_IMAGE)

        raw_body = await request.body()
        try:
            # JPEG decoding is CPU work too; keep it off the event loop so
            # /health remains responsive even for large images.
            image = await run_in_threadpool(decode_jpeg, raw_body)
        except InvalidImageError:
            return JSONResponse(status_code=400, content=INVALID_IMAGE)

        try:
            result = await run_in_threadpool(
                service.recognize,
                image.bgr,
                image.width,
                image.height,
            )
        except OCRNotReady:
            return JSONResponse(status_code=503, content=LOADING)
        except Exception:
            logger.exception("OCR request failed")
            return JSONResponse(status_code=500, content=OCR_FAILED)

        lines: list[dict[str, Any]] = [
            {
                "text": line.text,
                "score": line.score,
                "box": list(line.box),
            }
            for line in result.lines
        ]

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "width": image.width,
                "height": image.height,
                "ms": result.ms,
                "lines": lines,
            },
        )

    return app


app = create_app()
