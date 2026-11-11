"""End-to-end checks against a running KanjiOCRVibe service.

Start the service first, then run from the project root:
    .\.venv\Scripts\python.exe tests\integration_check.py

The script uses only the standard library and Pillow.  It generates test
images in memory and never writes them to disk.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\meiryo.ttc"),
    Path(r"C:\Windows\Fonts\YuGothM.ttc"),
    Path(r"C:\Windows\Fonts\msgothic.ttc"),
    Path(r"C:\Windows\Fonts\NotoSansJP-VF.ttf"),
]


def http(
    base_url: str,
    path: str,
    data: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 180.0,
) -> tuple[int, dict, float]:
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
            return response.status, body, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read())
        return exc.code, body, time.perf_counter() - started


def make_japanese_jpeg(width: int = 1400, height: int = 700) -> bytes:
    font_path = next((path for path in FONT_CANDIDATES if path.exists()), None)
    if font_path is None:
        raise RuntimeError("No Japanese font found in C:\\Windows\\Fonts")

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 64)
    lines = [
        "今日は雨が降る",
        "明日は晴れです",
        "東京駅で待っています",
        "日本語のOCRテスト",
    ]
    for index, text in enumerate(lines):
        draw.text((70, 60 + index * 140), text, font=font, fill="black")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def make_dense_jpeg(width: int = 2200, height: int = 1600) -> bytes:
    font_path = next((path for path in FONT_CANDIDATES if path.exists()), None)
    if font_path is None:
        raise RuntimeError("No Japanese font found in C:\\Windows\\Fonts")

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 44)
    for row in range(12):
        draw.text(
            (50, 40 + row * 125),
            f"{row + 1}行目 今日は雨が降る 明日は晴れです 日本語OCR",
            font=font,
            fill="black",
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def make_blank_jpeg(width: int = 900, height: int = 600) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def assert_health(body: dict, ready: bool) -> None:
    expected_keys = {"ok", "ready", "device", "model", "error"}
    assert set(body) == expected_keys, body
    assert body["ok"] is True, body
    assert body["ready"] is ready, body
    assert body["device"] == "cpu", body
    assert isinstance(body["model"], str) and body["model"], body
    assert body["error"] is None, body


def assert_ocr_shape(body: dict, width: int, height: int) -> None:
    assert body["ok"] is True, body
    assert body["width"] == width, body
    assert body["height"] == height, body
    assert isinstance(body["ms"], int) and body["ms"] >= 0, body
    assert isinstance(body["lines"], list), body

    previous_key = None
    for line in body["lines"]:
        assert set(line) == {"text", "score", "box"}, line
        assert isinstance(line["text"], str) and line["text"].strip(), line
        assert isinstance(line["score"], (int, float)), line
        assert 0.0 <= float(line["score"]) <= 1.0, line
        box = line["box"]
        assert isinstance(box, list) and len(box) == 4, line
        assert all(isinstance(value, int) for value in box), line
        x1, y1, x2, y2 = box
        assert 0 <= x1 <= x2 <= width, line
        assert 0 <= y1 <= y2 <= height, line
        key = (y1, x1)
        if previous_key is not None:
            assert key >= previous_key, (previous_key, key)
        previous_key = key


def project_files() -> set[str]:
    result: set[str] = set()
    for current, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".venv", "__pycache__"}]
        for name in files:
            result.add(os.path.relpath(os.path.join(current, name), ROOT))
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    print(f"Service: {base_url}")
    status, health, health_ms = http(base_url, "/health", timeout=5)
    print(f"initial /health: HTTP {status} in {health_ms:.0f} ms -> {json.dumps(health, ensure_ascii=False)}")
    assert status == 200, status

    deadline = time.monotonic() + 240
    while not health.get("ready"):
        assert health.get("ok") is True, health
        if time.monotonic() > deadline:
            raise AssertionError(f"model did not become ready: {health}")
        time.sleep(0.25)
        status, health, health_ms = http(base_url, "/health", timeout=5)
        assert status == 200, status

    assert_health(health, ready=True)
    print(f"ready /health: {json.dumps(health, ensure_ascii=False)}")

    japanese = make_japanese_jpeg()
    blank = make_blank_jpeg()

    files_before = project_files()

    status, body, elapsed = http(base_url, "/ocr", japanese, "image/jpeg")
    print(f"POST /ocr (Japanese): HTTP {status} in {elapsed:.2f}s, "
          f"{len(body.get('lines', []))} line(s), model ms={body.get('ms')}")
    assert status == 200, (status, body)
    assert_ocr_shape(body, 1400, 700)
    assert len(body["lines"]) >= 1, body
    print("  text:", " | ".join(line["text"] for line in body["lines"]))
    print("  boxes:", [line["box"] for line in body["lines"]])

    status, body, elapsed = http(base_url, "/ocr", blank, "image/jpeg")
    print(f"POST /ocr (blank): HTTP {status} in {elapsed:.2f}s, lines={body.get('lines')}")
    assert status == 200, (status, body)
    assert_ocr_shape(body, 900, 600)
    assert body["lines"] == [], body

    for payload in (b"", b"this is not a JPEG", b"\xff\xd8\xff\x00garbage"):
        status, body, _ = http(base_url, "/ocr", payload, "image/jpeg")
        print(f"POST /ocr (invalid {len(payload)} bytes): HTTP {status} -> {body}")
        assert status == 400, (status, body)
        assert body == {"ok": False, "error": "invalid image"}, body

    # Health must remain responsive while a dense OCR request is running.
    dense = make_dense_jpeg()
    dense_result: dict[str, object] = {}

    def run_dense():
        try:
            dense_result["response"] = http(base_url, "/ocr", dense, "image/jpeg")
        except BaseException as exc:  # pragma: no cover - diagnostic
            dense_result["error"] = exc

    worker = threading.Thread(target=run_dense)
    worker.start()
    time.sleep(0.1)
    health_latencies: list[float] = []
    while worker.is_alive():
        status, health_during, latency = http(base_url, "/health", timeout=2)
        assert status == 200, status
        assert_health(health_during, ready=True)
        health_latencies.append(latency)
        time.sleep(0.05)
    worker.join(timeout=5)
    assert "error" not in dense_result, dense_result
    status, body, elapsed = dense_result["response"]
    assert status == 200, (status, body)
    assert_ocr_shape(body, 2200, 1600)
    print(
        f"health during dense OCR: {len(health_latencies)} checks, "
        f"max latency={max(health_latencies or [0]) * 1000:.0f} ms"
    )
    # The first sample can coincide with the client connecting and with
    # PaddleOCR's one-time warm-up, so allow up to one second per health check.
    assert max(health_latencies or [0]) < 1.0, health_latencies

    # Two overlapping requests: both must succeed; serialization itself is
    # asserted by tests/test_service.py using an instrumented fake model.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(http, base_url, "/ocr", japanese, "image/jpeg") for _ in range(2)]
        results = [future.result() for future in futures]
    for status, body, elapsed in results:
        assert status == 200, (status, body)
        assert_ocr_shape(body, 1400, 700)
    print("two overlapping /ocr requests: both HTTP 200")

    files_after = project_files()
    created = sorted(files_after - files_before)
    assert not created, f"service wrote files into the project: {created}"
    print("no files written to the project by OCR requests")

    print("ALL INTEGRATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
