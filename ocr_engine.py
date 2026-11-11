"""PaddleOCR loading and inference for the local KanjiOCRVibe service.

The PaddleOCR import is intentionally deferred until the background loader runs,
so the web server can bind and answer health checks immediately.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Keep Paddle's C++ logging quiet before Paddle is imported.  PaddleOCR's own
# Python logging is configured below.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("FLAGS_minloglevel", "2")

logger = logging.getLogger("kanjiocrvibe.ocr_engine")

MODEL_NAME = "PP-OCRv4 (japan)"
DEVICE = "cpu"


class OCRNotReady(RuntimeError):
    """Raised when inference is requested before the model is ready."""


class OCRRuntimeError(RuntimeError):
    """Raised when PaddleOCR raises during inference."""


@dataclass(frozen=True)
class OCRLine:
    text: str
    score: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class OCRResult:
    ms: int
    lines: list[OCRLine]


@dataclass(frozen=True)
class EngineState:
    ready: bool
    device: str
    model: str
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "ready": self.ready,
            "device": self.device,
            "model": self.model,
            "error": self.error,
        }


@contextlib.contextmanager
def _capture_output():
    """Capture stdout/stderr while PaddleOCR is initializing or downloading."""

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        yield stream


def _format_error(exc: BaseException) -> str:
    detail = " ".join(str(exc).split())
    if detail:
        return f"{type(exc).__name__}: {detail}"[:300]
    return type(exc).__name__


def _is_writable_dir(directory: Path) -> bool:
    """Return whether a temporary file can actually be created in directory."""

    probe = directory / f".write-test-{os.getpid()}-{threading.get_ident()}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _configure_paddle_cache() -> None:
    """Keep Paddle's caches writable, falling back to a workspace cache.

    Paddle creates ``~/.cache/paddle`` during import and PaddleOCR downloads
    models under ``~/.paddleocr``.  A sandboxed or read-only user profile makes
    model loading fail.  In that case, redirect both Paddle caches into
    ``model_cache/`` next to this file before Paddle is imported.
    """

    home = Path.home()
    home_cache_writable = _is_writable_dir(home / ".cache" / "paddle")

    explicit_model_dir = os.environ.get("PADDLE_OCR_BASE_DIR")
    if explicit_model_dir:
        model_dir = Path(explicit_model_dir).expanduser()
        model_dir_writable = _is_writable_dir(model_dir)
    else:
        model_dir = home / ".paddleocr"
        model_dir_writable = _is_writable_dir(model_dir)

    if home_cache_writable and model_dir_writable:
        return

    local_root = Path(__file__).resolve().parent / "model_cache"
    local_home = local_root / "home"
    local_models = local_root / "paddleocr"
    if not _is_writable_dir(local_home / ".cache" / "paddle") or not _is_writable_dir(local_models):
        # Let Paddle raise its normal error; /health will report it.
        return

    # USERPROFILE/HOME affect Paddle's import-time dataset cache.  PaddleOCR
    # reads PADDLE_OCR_BASE_DIR when its module is imported.
    os.environ["USERPROFILE"] = str(local_home)
    os.environ["HOME"] = str(local_home)
    os.environ["PADDLE_OCR_BASE_DIR"] = str(local_models)
    logger.warning(
        "User Paddle cache is not writable; using workspace cache at %s",
        local_root,
    )


def _build_paddleocr() -> Any:
    """Import and construct PaddleOCR on the current (worker) thread."""

    _configure_paddle_cache()

    # Set these before the import as well as after, because both Paddle and
    # PaddleOCR install logging handlers during import.
    logging.getLogger("paddleocr").setLevel(logging.ERROR)
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    with _capture_output() as captured:
        from paddleocr import PaddleOCR  # imported lazily on the loader thread

        logging.getLogger("paddleocr").setLevel(logging.ERROR)
        logging.getLogger("ppocr").setLevel(logging.ERROR)

        try:
            model = PaddleOCR(
                use_angle_cls=True,  # angle classification for rotated text
                lang="japan",
                use_gpu=False,
                show_log=False,
            )
        except Exception:
            output = captured.getvalue().strip()
            if output:
                logger.error("PaddleOCR initialization output:\n%s", output[-4000:])
            raise

    return model


class OCREngine:
    """Owns the model and serializes all inference calls.

    ``loader`` is injectable so tests can exercise the HTTP layer without
    downloading or running the real model.
    """

    def __init__(self, loader: Callable[[], Any] | None = None) -> None:
        self._loader = loader or _build_paddleocr
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._model: Any | None = None
        self._ready = False
        self._error: str | None = None
        self._load_thread: threading.Thread | None = None

    def start_loading(self) -> None:
        """Start the one-time model load on a daemon thread.

        This never waits for PaddleOCR, so the HTTP server can start serving
        /health immediately.
        """

        with self._state_lock:
            if self._load_thread is not None:
                return
            self._error = None
            self._ready = False
            thread = threading.Thread(
                target=self._load,
                name="paddleocr-loader",
                daemon=True,
            )
            self._load_thread = thread

        try:
            thread.start()
        except BaseException as exc:
            with self._state_lock:
                self._error = _format_error(exc)
            logger.exception("Could not start the PaddleOCR loader thread")

    def _load(self) -> None:
        logger.info("Loading PaddleOCR model %s on CPU", MODEL_NAME)
        try:
            model = self._loader()
        except BaseException as exc:
            message = _format_error(exc)
            with self._state_lock:
                self._model = None
                self._ready = False
                self._error = message
            logger.exception("PaddleOCR model failed to load")
            return

        with self._state_lock:
            self._model = model
            self._ready = True
            self._error = None
        logger.info("PaddleOCR model is ready")

    def state(self) -> EngineState:
        with self._state_lock:
            return EngineState(
                ready=self._ready,
                device=DEVICE,
                model=MODEL_NAME,
                error=self._error,
            )

    def recognize(
        self,
        image: np.ndarray,
        width: int,
        height: int,
    ) -> OCRResult:
        """Run one OCR inference and convert Paddle's polygons to boxes.

        A single lock covers the PaddleOCR call, so overlapping HTTP requests
        queue here instead of invoking the model concurrently.
        """

        with self._inference_lock:
            with self._state_lock:
                model = self._model
                ready = self._ready

            if not ready or model is None:
                raise OCRNotReady("model is not ready")

            started = time.perf_counter()
            try:
                raw_result = model.ocr(image, cls=True)
            except Exception as exc:
                logger.exception("PaddleOCR inference failed")
                raise OCRRuntimeError("ocr failed") from exc

            lines = _parse_result(raw_result, width, height)
            elapsed_ms = int(round((time.perf_counter() - started) * 1000))

        return OCRResult(ms=max(0, elapsed_ms), lines=lines)


def _parse_result(
    raw_result: Any,
    width: int,
    height: int,
) -> list[OCRLine]:
    """Convert PaddleOCR 2.x output into API lines.

    PaddleOCR returns one page for one image:
        [[[polygon, (text, score)], ...]]

    A blank image can return ``None`` or an empty page.
    """

    if raw_result is None:
        return []

    if not isinstance(raw_result, (list, tuple)):
        raise OCRRuntimeError("unexpected PaddleOCR result type")

    if len(raw_result) == 0:
        return []

    page = raw_result
    if len(raw_result) == 1 and isinstance(raw_result[0], (list, tuple)):
        page = raw_result[0]

    if page is None:
        return []
    if not isinstance(page, (list, tuple)):
        raise OCRRuntimeError("unexpected PaddleOCR page type")

    lines: list[OCRLine] = []
    for entry in page:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue

        polygon, recognition = entry[0], entry[1]
        if not isinstance(recognition, (list, tuple)) or len(recognition) < 2:
            continue

        text = recognition[0]
        if text is None:
            continue
        text = str(text)
        if not text.strip():
            continue

        try:
            score = float(recognition[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue

        box = _polygon_to_box(polygon, width, height)
        if box is None:
            continue

        lines.append(OCRLine(text=text, score=score, box=box))

    # Explicit reading order required by the API: top-to-bottom, then
    # left-to-right for boxes on the same line.
    lines.sort(key=lambda line: (line.box[1], line.box[0], line.box[3], line.box[2]))
    return lines


def _polygon_to_box(
    polygon: Any,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Return the axis-aligned, image-clamped box for a PaddleOCR polygon."""

    try:
        points = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if points.size < 4 or points.size % 2 != 0:
        return None
    try:
        points = points.reshape(-1, 2)
    except ValueError:
        return None

    if points.shape[0] < 2 or not np.all(np.isfinite(points)):
        return None

    x1 = max(0, min(width, int(math.floor(float(points[:, 0].min())))))
    y1 = max(0, min(height, int(math.floor(float(points[:, 1].min())))))
    x2 = max(0, min(width, int(math.ceil(float(points[:, 0].max())))))
    y2 = max(0, min(height, int(math.ceil(float(points[:, 1].max())))))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return (x1, y1, x2, y2)
