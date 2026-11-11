# KanjiApp OCR

KanjiApp OCR is the standalone local Japanese OCR companion service for
[KanjiApp](https://github.com/andres-foo/kanji-app). KanjiApp owns the
screenshot queue, study data, UI, and kanji features; this repository provides
only the OCR backend.

It runs PaddleOCR behind a small HTTP API: raw JPEG bytes in, recognized
Japanese text lines with confidence scores and axis-aligned bounding boxes out.
It does not import KanjiApp, access its database, or otherwise depend on it, so
the OCR service can be run and tested independently. KanjiApp connects to it
over `http://127.0.0.1:8765`.

The service is intentionally narrow:

- no UI, translation, kanji lookup, database, or screenshot storage
- no cloud OCR calls, and uploaded images are never written to disk
- loopback-only binding (`127.0.0.1`), CPU inference
- one PaddleOCR inference at a time, with health checks staying responsive

## Tested stack

Tested on Windows with Python 3.11.9. The pinned dependency set targets
Python 3.11.

| Package | Tested version |
| --- | --- |
| Python | 3.11.9 |
| paddlepaddle (CPU) | 2.6.2 |
| paddleocr | 2.10.0 |
| fastapi | 0.115.6 |
| uvicorn | 0.34.0 |
| numpy | 1.26.4 |
| Pillow | 13.0.0 |
| opencv-python | 4.10.0.84 |
| opencv-contrib-python | 4.10.0.84 |
| opencv-python-headless | 4.10.0.84 |

PaddleOCR 2.10.0 is configured with `lang="japan"`, `use_gpu=False`, and
`use_angle_cls=True`. PaddleOCR uses the PP-OCRv4 Japanese recognition model
(`japan_PP-OCRv4_rec_infer`), the PP-OCRv3 multilingual detection model
(`Multilingual_PP-OCRv3_det_infer`), and the PP-OCRv2 angle classifier
(`ch_ppocr_mobile_v2.0_cls_infer`). The service reports the model name as
`PP-OCRv4 (japan)`.

## Setup (Windows)

Open PowerShell in this directory.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, use the virtual environment's
Python executable directly, as in the commands below.

Do not replace the pinned NumPy version with NumPy 2: PaddlePaddle 2.6.2 and
the pinned OpenCV 4.10 wheels require NumPy 1.26.4.

### Japanese model download and cache

The model is downloaded by PaddleOCR on first startup, not by this service.
The first launch therefore needs internet access and can take a few minutes.
The download is cached on the local machine, normally under:

```text
C:\Users\<you>\.paddleocr\whl\det\ml\Multilingual_PP-OCRv3_det_infer\
C:\Users\<you>\.paddleocr\whl\rec\japan\japan_PP-OCRv4_rec_infer\
C:\Users\<you>\.paddleocr\whl\cls\ch_ppocr_mobile_v2.0_cls_infer\
```

If the user profile is not writable (for example, a sandboxed launcher), the
service automatically falls back to a workspace cache under `model_cache\`
next to the app. Normal OCR operation is still local-only after the models are
downloaded.

To pre-download without starting the HTTP server, run a one-off constructor in
the activated environment:

```powershell
python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='japan', use_angle_cls=True, use_gpu=False, show_log=False)"
```

To use a different cache location, set `PADDLE_OCR_BASE_DIR` before startup.
The pinned PaddleOCR reads that variable and otherwise uses
`%USERPROFILE%\.paddleocr` by default.

## Start

Exact start command:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8765
```

Convenience options:

```powershell
.\run.ps1                 # same command, loopback-only, port 8765
.\run.ps1 -Port 8770      # alternate port, still loopback-only
```

Or double-click `start.bat`. It starts Uvicorn in the console window; closing
that window or pressing Ctrl+C stops the service.

Uvicorn binds quickly. PaddleOCR loads in a background daemon thread, so
`/health` answers `ready:false` while the model is still loading. The first
request that requires OCR while loading gets HTTP 503.

## HTTP API

Base URL: `http://127.0.0.1:8765`

### GET /health

Always HTTP 200. While loading:

```json
{
  "ok": true,
  "ready": false,
  "device": "cpu",
  "model": "PP-OCRv4 (japan)",
  "error": null
}
```

When ready, `ready` is `true`. If loading fails, HTTP is still 200 and the body
is:

```json
{
  "ok": false,
  "ready": false,
  "device": "cpu",
  "model": "PP-OCRv4 (japan)",
  "error": "RuntimeError: short failure detail"
}
```

curl example:

```powershell
curl.exe http://127.0.0.1:8765/health
```

### POST /ocr

- body: raw JPEG bytes (no multipart, base64, or JSON wrapper)
- required header: `Content-Type: image/jpeg`
- returns decoded image dimensions, OCR elapsed milliseconds, and lines

Success:

```json
{
  "ok": true,
  "width": 1280,
  "height": 720,
  "ms": 940,
  "lines": [
    {
      "text": "今日は雨が降る",
      "score": 0.99,
      "box": [120, 80, 640, 160]
    }
  ]
}
```

`box` is `[x1, y1, x2, y2]` for the axis-aligned bounding box of the
PaddleOCR polygon. Lines are sorted top-to-bottom, then left-to-right. An image
with no text returns `lines: []`.

Failure responses:

| Status | Body |
| --- | --- |
| 400 | `{"ok": false, "error": "invalid image"}` |
| 500 | `{"ok": false, "error": "ocr failed"}` |
| 503 | `{"ok": false, "error": "loading"}` |

curl example using raw JPEG bytes:

```powershell
curl.exe -X POST "http://127.0.0.1:8765/ocr" `
  -H "Content-Type: image/jpeg" `
  --data-binary "@C:\path\to\japanese.jpg"
```

The service supports EXIF orientation, so the reported width, height, and
boxes correspond to the image as displayed. JPEG decoding and PaddleOCR
inference both run in worker threads. Inference is serialized by a lock.

Images are held in memory only and are not persisted. The service disables
FastAPI's documentation UI; it has no other endpoints.

## Tests

Offline contract/concurrency tests (fake model, no download needed):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

End-to-end check against a running service (real Japanese model):

```powershell
# terminal 1
.\run.ps1

# terminal 2, after the service reports ready
.\.venv\Scripts\python.exe tests\integration_check.py
```

The integration check generates Japanese and blank JPEGs in memory, verifies
the response contract and box bounds, tests invalid input, checks health during
a dense OCR request, sends two overlapping OCR requests, and confirms that
request handling creates no files in the project directory.

## Files

```text
app.py                  FastAPI app and HTTP endpoints
ocr_engine.py           background model loading, serialized inference, box conversion
requirements.txt        exact tested dependency versions
run.ps1                 PowerShell start script
start.bat               double-click start; closing the console stops the service
tests/                  offline HTTP tests and live integration check
```

## Notes and limits

- This service has no authentication and is intended for local use only. It
  binds to `127.0.0.1`; do not expose it on a network interface.
- There is no request-size limit beyond available memory. The client should
  send a reasonably sized screenshot.
- PaddleOCR's own logging is suppressed; model-download and initialization
  output is captured rather than printed to the service console.
- `ms` measures the serialized OCR call (inference plus polygon-to-box
  conversion) and excludes waiting for an earlier request and JPEG decoding.
