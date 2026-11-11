"""Offline HTTP contract and concurrency tests.

Run from the project root:
    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

These tests use a fake PaddleOCR model, so they do not need the downloaded
Japanese model.  The real-model check is tests/integration_check.py.
"""

from __future__ import annotations

import io
import json
import os
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn
from PIL import Image, ImageDraw

from app import create_app, decode_jpeg
from ocr_engine import OCREngine, _parse_result


def make_jpeg(width: int = 640, height: int = 360, blank: bool = False) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    if not blank:
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 300, 100), outline="black", width=3)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class SlowFakePaddleOCR:
    """PaddleOCR-shaped fake that records overlapping inference calls."""

    def __init__(self, delay: float = 0.4) -> None:
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def ocr(self, image, cls: bool = True):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            call_number = self.calls
        try:
            time.sleep(self.delay)
            # Almost-white image => PaddleOCR-style empty result.
            if int(image.min()) >= 245:
                return [None]
            return [
                [
                    [
                        [[20.0, 20.0], [200.0, 20.0], [200.0, 60.0], [20.0, 60.0]],
                        ("今日は雨が降る", 0.99),
                    ],
                    [
                        [[20.0, 100.0], [240.0, 100.0], [240.0, 140.0], [20.0, 140.0]],
                        ("明日は晴れ", 0.95),
                    ],
                ]
            ]
        finally:
            with self._lock:
                self.active -= 1


class FakeEngineApp:
    def __init__(self, delay: float = 0.4, model=None) -> None:
        self.model = model if model is not None else SlowFakePaddleOCR(delay=delay)
        self.load_gate = threading.Event()

        def loader():
            # Deterministic "still loading" state until the test opens the gate.
            if not self.load_gate.wait(timeout=10):
                raise RuntimeError("test load gate timed out")
            return self.model

        self.engine = OCREngine(loader=loader)
        self.app = create_app(self.engine)
        self.server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host="127.0.0.1",
                port=free_port(),
                log_level="warning",
                access_log=False,
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.base_url = f"http://127.0.0.1:{self.server.config.port}"

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self.server.started:
            raise RuntimeError("test server did not start")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.should_exit = True
        self.thread.join(timeout=10)

    def request(self, path: str, data: bytes | None = None, content_type: str | None = None, timeout: float = 10):
        headers = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def wait_ready(self, timeout: float = 10) -> dict:
        self.load_gate.set()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, body = self.request("/health")
            if status == 200 and body.get("ready"):
                return body
            time.sleep(0.05)
        raise AssertionError("model did not become ready")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServiceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FakeEngineApp(delay=0.4).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)

    def test_01_health_is_immediate_while_loading(self):
        started = time.perf_counter()
        status, body = self.server.request("/health")
        elapsed = time.perf_counter() - started
        self.assertEqual(status, 200)
        self.assertEqual(body["device"], "cpu")
        self.assertEqual(body["model"], "PP-OCRv4 (japan)")
        self.assertIsNone(body["error"])
        self.assertTrue(body["ok"])
        self.assertLess(elapsed, 0.5)
        # The loader sleeps for 1 second. Unless this test was delayed, health
        # must still be serving while the model is loading.
        self.assertFalse(body["ready"])

    def test_02_ocr_not_ready_returns_503(self):
        status, body = self.server.request(
            "/ocr", make_jpeg(), "image/jpeg"
        )
        self.assertEqual(status, 503)
        self.assertEqual(body, {"ok": False, "error": "loading"})

    def test_03_ready_health(self):
        body = self.server.wait_ready()
        self.assertEqual(
            body,
            {
                "ok": True,
                "ready": True,
                "device": "cpu",
                "model": "PP-OCRv4 (japan)",
                "error": None,
            },
        )

    def test_04_valid_jpeg_returns_ordered_lines_and_boxes(self):
        self.server.wait_ready()
        status, body = self.server.request(
            "/ocr", make_jpeg(), "image/jpeg"
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual((body["width"], body["height"]), (640, 360))
        self.assertIsInstance(body["ms"], int)
        self.assertGreaterEqual(body["ms"], 0)
        self.assertEqual(len(body["lines"]), 2)
        for line in body["lines"]:
            self.assertEqual(set(line), {"text", "score", "box"})
            self.assertIsInstance(line["text"], str)
            self.assertTrue(line["text"])
            self.assertIsInstance(line["score"], float)
            self.assertEqual(len(line["box"]), 4)
            x1, y1, x2, y2 = line["box"]
            for value in line["box"]:
                self.assertIsInstance(value, int)
            self.assertTrue(0 <= x1 <= x2 <= body["width"])
            self.assertTrue(0 <= y1 <= y2 <= body["height"])
        self.assertEqual(
            [line["text"] for line in body["lines"]],
            ["今日は雨が降る", "明日は晴れ"],
        )

    def test_05_blank_image_returns_empty_lines(self):
        self.server.wait_ready()
        status, body = self.server.request(
            "/ocr", make_jpeg(blank=True), "image/jpeg"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["lines"], [])
        self.assertEqual((body["width"], body["height"]), (640, 360))

    def test_06_invalid_bytes_return_400(self):
        self.server.wait_ready()
        for payload in (b"", b"not an image", b"\xff\xd8\xff" + b"broken" * 20):
            status, body = self.server.request("/ocr", payload, "image/jpeg")
            self.assertEqual(status, 400, payload[:20])
            self.assertEqual(body, {"ok": False, "error": "invalid image"})

    def test_07_non_jpeg_content_type_returns_400(self):
        self.server.wait_ready()
        status, body = self.server.request(
            "/ocr", make_jpeg(), "image/png"
        )
        self.assertEqual(status, 400)
        self.assertEqual(body, {"ok": False, "error": "invalid image"})

    def test_08_overlapping_requests_are_serialized_and_health_stays_up(self):
        self.server.wait_ready()
        before = self.server.model.calls
        payload = make_jpeg()

        results: list[tuple[int, dict]] = []
        errors: list[BaseException] = []

        def worker():
            try:
                results.append(self.server.request("/ocr", payload, "image/jpeg"))
            except BaseException as exc:  # pragma: no cover - diagnostic
                errors.append(exc)

        first = threading.Thread(target=worker)
        second = threading.Thread(target=worker)
        first.start()
        second.start()

        time.sleep(0.12)
        started = time.perf_counter()
        health_status, health_body = self.server.request("/health", timeout=2)
        health_elapsed = time.perf_counter() - started
        self.assertEqual(health_status, 200)
        self.assertTrue(health_body["ready"])
        self.assertLess(health_elapsed, 0.5, "health blocked behind OCR")

        first.join(timeout=10)
        second.join(timeout=10)
        self.assertFalse(errors)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(status == 200 for status, _ in results))
        self.assertEqual(self.server.model.calls - before, 2)
        self.assertEqual(
            self.server.model.max_active,
            1,
            "PaddleOCR inference ran concurrently",
        )


class ImageAndParsingTests(unittest.TestCase):
    def test_exif_orientation_updates_dimensions(self):
        image = Image.new("RGB", (120, 40), "white")
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        decoded = decode_jpeg(buffer.getvalue())
        self.assertEqual((decoded.width, decoded.height), (40, 120))
        self.assertEqual(decoded.bgr.shape, (120, 40, 3))

    def test_polygons_are_clamped_and_sorted(self):
        raw = [
            [
                [
                    [[-10, -10], [200, -10], [200, 30], [-10, 30]],
                    ("first", 0.9),
                ],
                [
                    [[5, 50], [90, 50], [90, 80], [5, 80]],
                    ("second", 0.8),
                ],
            ]
        ]
        lines = _parse_result(raw, 100, 60)
        self.assertEqual([line.text for line in lines], ["first", "second"])
        self.assertEqual(lines[0].box, (0, 0, 100, 30))
        self.assertEqual(lines[1].box, (5, 50, 90, 60))

    def test_blank_paddle_results(self):
        self.assertEqual(_parse_result(None, 10, 10), [])
        self.assertEqual(_parse_result([], 10, 10), [])
        self.assertEqual(_parse_result([None], 10, 10), [])
        self.assertEqual(_parse_result([[]], 10, 10), [])


class FailingPaddleOCR:
    def ocr(self, image, cls: bool = True):
        raise RuntimeError("synthetic inference failure")


class RuntimeFailureTests(unittest.TestCase):
    def test_inference_failure_returns_500_contract_body(self):
        with FakeEngineApp(model=FailingPaddleOCR()) as server:
            server.wait_ready()
            status, body = server.request("/ocr", make_jpeg(), "image/jpeg")
        self.assertEqual(status, 500)
        self.assertEqual(body, {"ok": False, "error": "ocr failed"})


class FailedLoadTests(unittest.TestCase):
    def test_system_exit_from_loader_is_reported_as_an_error(self):
        engine = OCREngine(loader=lambda: (_ for _ in ()).throw(SystemExit(2)))
        engine.start_loading()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = engine.state()
            if state.error:
                break
            time.sleep(0.02)
        self.assertFalse(state.ok)
        self.assertFalse(state.ready)
        self.assertIn("SystemExit", state.error)

    def test_failed_model_load_is_reported_but_health_stays_200(self):
        def broken_loader():
            raise RuntimeError("synthetic load failure")

        engine = OCREngine(loader=broken_loader)
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(engine),
                host="127.0.0.1",
                port=free_port(),
                log_level="critical",
                access_log=False,
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            body = None
            while time.monotonic() < deadline:
                if server.started:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{server.config.port}/health", timeout=2
                    ) as response:
                        status = response.status
                        body = json.loads(response.read())
                    if body.get("error"):
                        break
                time.sleep(0.05)
            self.assertEqual(status, 200)
            self.assertFalse(body["ok"])
            self.assertFalse(body["ready"])
            self.assertEqual(body["device"], "cpu")
            self.assertIn("synthetic load failure", body["error"])
        finally:
            server.should_exit = True
            thread.join(timeout=10)


def project_image_files(root: Path) -> set[Path]:
    found: set[Path] = set()
    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in {".venv", "__pycache__"}]
        for name in files:
            path = Path(current) / name
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                found.add(path)
    return found


class NoUploadedImageFilesTests(unittest.TestCase):
    def test_service_creates_no_image_files(self):
        root = Path(__file__).resolve().parents[1]
        before = project_image_files(root)
        with FakeEngineApp(delay=0.05) as server:
            server.wait_ready()
            status, _ = server.request("/ocr", make_jpeg(), "image/jpeg")
            self.assertEqual(status, 200)
        after = project_image_files(root)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
