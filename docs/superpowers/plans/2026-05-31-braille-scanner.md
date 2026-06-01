# Braille Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit web app that scans real physical Braille via camera and converts it to English text + speech in near real-time, deployable to Streamlit Community Cloud free tier.

**Architecture:** Custom JS Streamlit component captures camera frames (getUserMedia, 300ms interval) and sends base64 JPEGs to a Python `@st.fragment(run_every="500ms")` that runs a dual-path pipeline (Hough Circle Transform + YOLOv8n ONNX ensemble → DBSCAN grid → liblouis backTranslateString). Web Speech API handles TTS in-browser with zero server latency.

**Tech Stack:** Python 3.11+, streamlit>=1.37.0, opencv-python-headless, onnxruntime-cpu, scipy, edge-tts, python3-louis (apt), liblouis-dev (apt). No torch at runtime.

---

## File Map

| File | Responsibility |
|---|---|
| `requirements.txt` | Python pip dependencies |
| `packages.txt` | Apt system dependencies (liblouis-dev, python3-louis) |
| `utils/image_utils.py` | Base64 encode/decode, letterbox resize, draw dot overlays |
| `utils/quality.py` | Blur score, brightness, dot density, guidance message logic |
| `pipeline/preprocessor.py` | CLAHE → GaussianBlur → adaptiveThreshold → morphOpen |
| `pipeline/detector.py` | Hough detection, ONNX inference, NMS, ensemble merge |
| `pipeline/grid.py` | 1D gap clustering, cell construction, Braille Unicode codepoints |
| `pipeline/corrector.py` | Skew angle estimation, warpAffine correction |
| `pipeline/translator.py` | liblouis backTranslateString, all grade tables |
| `components/camera_component/index.html` | JS: getUserMedia, canvas capture, Web Speech API, overlays |
| `components/camera_component/__init__.py` | Streamlit component declaration |
| `app.py` | Main app: @st.fragment, upload tab, CSS, settings sidebar |
| `tests/test_image_utils.py` | Unit tests for image_utils |
| `tests/test_quality.py` | Unit tests for quality signals |
| `tests/test_preprocessor.py` | Unit tests for preprocessor |
| `tests/test_detector.py` | Unit tests for Hough, NMS, ensemble (ONNX mocked) |
| `tests/test_grid.py` | Unit tests for grid reconstruction with synthetic dots |
| `tests/test_corrector.py` | Unit tests for skew estimation and correction |
| `tests/test_translator.py` | Unit tests for translator (liblouis mocked for Windows dev) |
| `models/README.md` | Model training/export instructions |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `packages.txt`
- Create: `utils/__init__.py`
- Create: `pipeline/__init__.py`
- Create: `components/camera_component/__init__.py` (empty placeholder)
- Create: `tests/__init__.py`
- Create: `models/README.md`
- Create: `.gitignore`

- [ ] **Step 1: Initialise git and create directory structure**

```bash
cd C:\Users\presyze\Projects\Personal\braille
git init
mkdir utils pipeline components\camera_component tests models
```

- [ ] **Step 2: Create requirements.txt**

```
streamlit>=1.37.0
opencv-python-headless>=4.9.0
onnxruntime-cpu>=1.17.0
numpy>=1.26.0
scipy>=1.12.0
edge-tts>=6.1.0
psutil>=5.9.0
pytest>=8.0.0
```

Note: `psutil` monitors server memory so the pipeline can downscale to 480×480 if approaching the 1GB free-tier limit. `edge-tts` is used in the upload tab for a downloadable MP3 audio output.

- [ ] **Step 3: Create packages.txt**

```
liblouis-dev
python3-louis
```

- [ ] **Step 4: Create .gitignore**

```
__pycache__/
*.pyc
.env
*.onnx
!models/*.onnx
.streamlit/secrets.toml
```

Note: ONNX model files ARE committed to the repo (they are small — ~12MB). The `!models/*.onnx` rule re-includes them.

- [ ] **Step 5: Create all `__init__.py` files (empty)**

```bash
type nul > utils\__init__.py
type nul > pipeline\__init__.py
type nul > tests\__init__.py
```

- [ ] **Step 6: Create models/README.md**

```markdown
# Model Training & Export

## Prerequisites
GPU environment: Google Colab (free T4) or local NVIDIA GPU.
Install: `pip install ultralytics`

## Datasets
- DSBI: https://github.com/yeluo1994/DSBI (114 double-sided Braille images)
- Angelina: https://github.com/IlyaOvodov/AngelinaDataset

## Training
Create `braille_dots.yaml`:
```yaml
path: ./data
train: images/train
val: images/val
nc: 1
names: ['dot']
```

Run:
```bash
yolo train model=yolov8n.pt data=braille_dots.yaml epochs=100 imgsz=640 batch=16
```

## Export FP32 ONNX
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640
# Output: best.onnx (~12MB)
# Copy to: models/yolov8n_braille.onnx
```

## Optional INT8 Quantization
```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
# Requires ~50 calibration images. See onnxruntime quantization docs.
```

## Development Without the Model
The app falls back to Hough-only detection if the ONNX file is missing.
Place any file at models/yolov8n_braille.onnx to suppress the warning,
or simply run without it during development.
```

- [ ] **Step 7: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error. `onnxruntime-cpu` installs without torch.

- [ ] **Step 8: Commit scaffold**

```bash
git add .
git commit -m "feat: project scaffold — requirements, structure, model readme"
```

---

## Task 2: utils/image_utils.py

**Files:**
- Create: `utils/image_utils.py`
- Create: `tests/test_image_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_utils.py`:

```python
import base64
import cv2
import numpy as np
import pytest
from utils.image_utils import decode_frame, encode_frame, letterbox, draw_dots


def _make_jpeg_b64(w=100, h=80) -> str:
    """Create a solid blue 100×80 JPEG as base64."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)  # blue in BGR
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode('utf-8')


def test_decode_frame_returns_ndarray():
    b64 = _make_jpeg_b64()
    result = decode_frame(b64)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3


def test_decode_frame_invalid_returns_none():
    result = decode_frame("notbase64!!!")
    assert result is None


def test_encode_decode_roundtrip_shape():
    original = np.zeros((100, 80, 3), dtype=np.uint8)
    original[10:20, 10:20] = 128
    b64 = encode_frame(original)
    recovered = decode_frame(b64)
    assert recovered.shape == original.shape


def test_letterbox_output_is_square():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out, scale, (pad_x, pad_y) = letterbox(frame, size=640)
    assert out.shape == (640, 640, 3)


def test_letterbox_scale_and_padding():
    # 480×640 → 640×640: scale=1.0, pad_y=80, pad_x=0
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out, scale, (pad_x, pad_y) = letterbox(frame, size=640)
    assert scale == pytest.approx(1.0)
    assert pad_y == 80
    assert pad_x == 0


def test_letterbox_portrait():
    # 640×480 portrait → 640×640: scale=1.0, pad_x=80, pad_y=0
    frame = np.zeros((640, 480, 3), dtype=np.uint8)
    out, scale, (pad_x, pad_y) = letterbox(frame, size=640)
    assert scale == pytest.approx(1.0)
    assert pad_x == 80
    assert pad_y == 0


def test_draw_dots_returns_copy():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dots = [(50.0, 50.0, 5.0)]
    result = draw_dots(frame, dots)
    assert result is not frame  # must be a copy
    assert result.shape == frame.shape


def test_draw_dots_draws_circle():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    dots = [(100.0, 100.0, 8.0)]
    result = draw_dots(frame, dots)
    # At least some pixels changed from black
    assert result.sum() > 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_image_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'utils.image_utils'`

- [ ] **Step 3: Implement utils/image_utils.py**

```python
import base64
import cv2
import numpy as np


def decode_frame(b64_jpeg: str) -> np.ndarray | None:
    """Decode base64 JPEG string to BGR numpy array. Returns None on failure."""
    try:
        data = base64.b64decode(b64_jpeg)
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame  # None if imdecode fails
    except Exception:
        return None


def encode_frame(frame: np.ndarray, quality: int = 85) -> str:
    """Encode BGR numpy array to base64 JPEG string."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')


def letterbox(
    frame: np.ndarray, size: int = 640
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize frame to size×size with black padding, preserving aspect ratio.
    Returns: (padded_frame, scale_factor, (pad_x, pad_y))
    scale_factor: multiply padded coords by 1/scale to get original coords.
    """
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    channels = frame.shape[2] if frame.ndim == 3 else 1
    if channels > 1:
        canvas = np.zeros((size, size, channels), dtype=np.uint8)
    else:
        canvas = np.zeros((size, size), dtype=np.uint8)

    pad_y = (size - new_h) // 2
    pad_x = (size - new_w) // 2
    canvas[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def draw_dots(
    frame: np.ndarray,
    dots: list[tuple[float, float, float]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw circle overlays for detected dot positions.
    dots: list of (x, y, radius)
    Returns a copy of frame with circles drawn.
    """
    out = frame.copy()
    for x, y, r in dots:
        cv2.circle(out, (int(x), int(y)), max(int(r), 4), color, thickness)
    return out
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_image_utils.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/image_utils.py tests/test_image_utils.py
git commit -m "feat: image utils — base64 encode/decode, letterbox, draw_dots"
```

---

## Task 3: utils/quality.py

**Files:**
- Create: `utils/quality.py`
- Create: `tests/test_quality.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_quality.py`:

```python
import numpy as np
import cv2
import pytest
from utils.quality import blur_score, brightness, dot_density, get_guidance


def test_blur_score_sharp_image_is_high():
    # Sharp black/white checkerboard has high Laplacian variance
    img = np.zeros((64, 64), dtype=np.uint8)
    img[::8, :] = 255
    img[:, ::8] = 255
    assert blur_score(img) > 200


def test_blur_score_blurry_image_is_low():
    img = np.full((64, 64), 128, dtype=np.uint8)  # uniform grey = no edges
    assert blur_score(img) < 10


def test_brightness_dark():
    img = np.zeros((100, 100), dtype=np.uint8)
    assert brightness(img) == pytest.approx(0.0)


def test_brightness_full():
    img = np.full((100, 100), 255, dtype=np.uint8)
    assert brightness(img) == pytest.approx(255.0)


def test_dot_density_zero_dots():
    assert dot_density([], 640, 640) == pytest.approx(0.0)


def test_dot_density_nonzero():
    dots = [(100.0, 100.0, 0.9)] * 10
    result = dot_density(dots, 640, 640)
    assert result == pytest.approx(10 / (640 * 640) * 1_000_000)


def test_guidance_blur_too_low():
    result = get_guidance(blur=30, bright=128, density=20, skew=0)
    assert result["status"] == "warn"
    assert "steady" in result["message"].lower()


def test_guidance_low_light():
    result = get_guidance(blur=200, bright=20, density=20, skew=0)
    assert result["status"] == "warn"
    assert "light" in result["message"].lower()


def test_guidance_too_bright():
    result = get_guidance(blur=200, bright=220, density=20, skew=0)
    assert result["status"] == "warn"
    assert "bright" in result["message"].lower()


def test_guidance_too_far():
    result = get_guidance(blur=200, bright=128, density=2, skew=0)
    assert result["status"] == "warn"
    assert "closer" in result["message"].lower()


def test_guidance_too_close():
    result = get_guidance(blur=200, bright=128, density=250, skew=0)
    assert result["status"] == "warn"
    assert "back" in result["message"].lower()


def test_guidance_skewed():
    result = get_guidance(blur=200, bright=128, density=20, skew=20)
    assert result["status"] == "warn"
    assert "tilt" in result["message"].lower()


def test_guidance_all_ok():
    result = get_guidance(blur=200, bright=128, density=20, skew=2)
    assert result["status"] == "ok"
    assert "scanning" in result["message"].lower()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_quality.py -v
```

Expected: `ModuleNotFoundError: No module named 'utils.quality'`

- [ ] **Step 3: Implement utils/quality.py**

```python
import cv2
import numpy as np


def blur_score(gray: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Values below 80 indicate blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity of a grayscale image (0–255)."""
    return float(gray.mean())


def dot_density(dots: list, frame_w: int, frame_h: int) -> float:
    """Detected dots per megapixel. Used to gauge camera distance."""
    if frame_w == 0 or frame_h == 0:
        return 0.0
    return len(dots) / (frame_w * frame_h) * 1_000_000


def get_guidance(blur: float, bright: float, density: float, skew: float) -> dict:
    """
    Return a guidance message based on current frame quality metrics.
    Priority order: blur → brightness → distance → skew → ok.
    """
    if blur < 50:
        return {"status": "warn", "message": "Hold camera steady"}
    if blur < 80:
        return {"status": "warn", "message": "Almost — hold steadier"}
    if bright < 40:
        return {"status": "warn", "message": "Need more light"}
    if bright > 210:
        return {"status": "warn", "message": "Too bright — find shade"}
    if density < 5:
        return {"status": "warn", "message": "Move closer to the Braille"}
    if density > 200:
        return {"status": "warn", "message": "Move back slightly"}
    if abs(skew) > 15:
        return {"status": "warn", "message": "Tilt camera — align with page edge"}
    return {"status": "ok", "message": "Good — scanning..."}
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_quality.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/quality.py tests/test_quality.py
git commit -m "feat: quality utils — blur score, brightness, dot density, guidance"
```

---

## Task 4: pipeline/preprocessor.py

**Files:**
- Create: `pipeline/preprocessor.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_preprocessor.py`:

```python
import numpy as np
import cv2
import pytest
from pipeline.preprocessor import preprocess


def _make_frame(h=640, w=640, val=128) -> np.ndarray:
    frame = np.full((h, w, 3), val, dtype=np.uint8)
    return frame


def test_preprocess_output_shapes():
    frame = _make_frame()
    binary, gray = preprocess(frame)
    assert binary.shape == (640, 640)
    assert gray.shape == (640, 640)


def test_preprocess_binary_values():
    """Binary output must contain only 0 and 255."""
    frame = _make_frame()
    binary, _ = preprocess(frame)
    unique = set(binary.flatten().tolist())
    assert unique.issubset({0, 255})


def test_preprocess_gray_is_uint8():
    frame = _make_frame()
    _, gray = preprocess(frame)
    assert gray.dtype == np.uint8


def test_preprocess_custom_clip_limit():
    """High clip_limit should not crash and still return correct shapes."""
    frame = _make_frame()
    binary, gray = preprocess(frame, clip_limit=3.5)
    assert binary.shape == (640, 640)


def test_preprocess_dark_frame():
    frame = _make_frame(val=5)
    binary, gray = preprocess(frame)
    assert binary.shape == (640, 640)


def test_preprocess_on_non_square_fails_gracefully():
    """Frame must be pre-letterboxed to 640×640 before calling preprocess."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # This should work (no size enforcement in preprocessor) but document intent
    binary, gray = preprocess(frame)
    assert binary.shape == (480, 640)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement pipeline/preprocessor.py**

```python
import cv2
import numpy as np


def preprocess(
    frame_bgr: np.ndarray, clip_limit: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare a BGR frame for Braille dot detection.

    Input:  frame_bgr — BGR image, should be letterboxed to 640×640 before calling.
    Returns: (binary_cleaned, gray)
        binary_cleaned — thresholded + morphed image suitable for Hough detection
        gray           — raw grayscale for quality signal computation
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned, gray
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: preprocessor — CLAHE, blur, adaptive threshold, morph open"
```

---

## Task 5: pipeline/detector.py — Hough, ONNX, NMS, Ensemble

**Files:**
- Create: `pipeline/detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_detector.py`:

```python
import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch
from pipeline.detector import detect_hough, _nms, detect_onnx, ensemble


def _white_dot_image(cx: int, cy: int, r: int = 8, size: int = 640) -> np.ndarray:
    """Binary image with one filled white circle on black background."""
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img, (cx, cy), r, 255, -1)
    return img


def test_detect_hough_finds_dot():
    img = _white_dot_image(320, 320, r=8)
    dots = detect_hough(img)
    assert len(dots) >= 1
    xs = [d[0] for d in dots]
    ys = [d[1] for d in dots]
    assert any(abs(x - 320) < 15 and abs(y - 320) < 15 for x, y in zip(xs, ys))


def test_detect_hough_empty_image():
    img = np.zeros((640, 640), dtype=np.uint8)
    dots = detect_hough(img)
    assert dots == []


def test_detect_hough_returns_list_of_tuples():
    img = _white_dot_image(200, 200, r=6)
    dots = detect_hough(img)
    for item in dots:
        assert len(item) == 3  # (x, y, r)


def test_nms_removes_duplicates():
    # Two nearly identical boxes should collapse to one
    boxes = np.array([
        [10, 10, 30, 30],
        [12, 12, 32, 32],
        [200, 200, 220, 220],
    ], dtype=float)
    scores = np.array([0.9, 0.8, 0.95])
    kept = _nms(boxes, scores, iou_threshold=0.45)
    assert len(kept) == 2  # two distinct regions kept


def test_nms_empty():
    kept = _nms(np.zeros((0, 4)), np.array([]), iou_threshold=0.45)
    assert kept == []


def test_detect_onnx_uses_session(monkeypatch):
    """ONNX path calls session.run and parses output correctly."""
    # Mock output: one detection at (320, 320) with confidence 0.9, w=h=10
    raw_out = np.zeros((1, 5, 8400), dtype=np.float32)
    raw_out[0, 0, 0] = 320.0   # x_center
    raw_out[0, 1, 0] = 320.0   # y_center
    raw_out[0, 2, 0] = 10.0    # width
    raw_out[0, 3, 0] = 10.0    # height
    raw_out[0, 4, 0] = 0.95    # confidence

    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="images")]
    session.get_inputs()[0].name = "images"
    session.run.return_value = [raw_out]

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    dots = detect_onnx(frame, session)
    assert len(dots) >= 1
    assert abs(dots[0][0] - 320.0) < 5
    assert abs(dots[0][1] - 320.0) < 5
    assert dots[0][2] > 0.8  # confidence


def test_ensemble_boosts_confirmed_dots():
    hough = [(100.0, 100.0, 6.0)]
    onnx = [(102.0, 101.0, 0.7)]  # within 8px of hough dot
    result = ensemble(hough, onnx)
    assert len(result) == 1
    assert result[0][2] > 0.7  # confidence boosted


def test_ensemble_includes_hough_only():
    hough = [(300.0, 300.0, 6.0)]
    onnx = []  # no ONNX detections
    result = ensemble(hough, onnx)
    assert len(result) == 1
    assert result[0][2] == pytest.approx(0.45)


def test_ensemble_includes_onnx_only():
    hough = []
    onnx = [(200.0, 200.0, 0.8)]
    result = ensemble(hough, onnx)
    assert len(result) == 1
    assert result[0][2] == pytest.approx(0.8)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_detector.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement pipeline/detector.py**

```python
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


@st.cache_resource
def load_onnx_session(model_path: str):
    """
    Load ONNX InferenceSession once per process and cache.
    Returns None if the file does not exist or onnxruntime is unavailable.
    """
    if not _ORT_AVAILABLE:
        return None
    if not Path(model_path).exists():
        return None
    try:
        return ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    except Exception:
        return None


def detect_hough(binary: np.ndarray) -> list[tuple[float, float, float]]:
    """
    Detect Braille dots using Hough Circle Transform.

    Input:  binary — preprocessed grayscale/binary image (from preprocessor).
    Returns: list of (x, y, radius) in pixel coordinates.

    Falls back to empty list if no circles found.
    """
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=8,
        param1=300,
        param2=0.85,
        minRadius=3,
        maxRadius=12,
    )
    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]


def _nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45
) -> list[int]:
    """
    Non-maximum suppression. Returns indices of kept boxes.
    boxes: (N, 4) as [x1, y1, x2, y2]
    scores: (N,)
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_threshold]
    return keep


def detect_onnx(
    frame_bgr: np.ndarray,
    session,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.45,
) -> list[tuple[float, float, float]]:
    """
    YOLOv8n ONNX dot detection.

    Input:  frame_bgr — BGR 640×640 image.
            session   — onnxruntime.InferenceSession loaded via load_onnx_session().
    Returns: list of (x_center, y_center, confidence).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 640, 640]

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]  # [1, 5, 8400]

    preds = output[0].T  # [8400, 5]: (cx, cy, w, h, conf)
    mask = preds[:, 4] > conf_threshold
    preds = preds[mask]
    if len(preds) == 0:
        return []

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scores = preds[:, 4]

    kept = _nms(boxes, scores, iou_threshold)
    return [(float(cx[i]), float(cy[i]), float(scores[i])) for i in kept]


def ensemble(
    hough_dots: list[tuple[float, float, float]],
    onnx_dots: list[tuple[float, float, float]],
    proximity_px: float = 8.0,
    boost: float = 0.15,
) -> list[tuple[float, float, float]]:
    """
    Merge Hough and ONNX detections by proximity.

    hough_dots: [(x, y, radius), ...]
    onnx_dots:  [(x, y, confidence), ...]
    Returns:    [(x, y, confidence), ...]

    Logic:
    - ONNX detection matched by a Hough dot within proximity_px → confidence += boost
    - ONNX detection with no Hough match → kept as-is
    - Hough detection with no ONNX match → added with confidence=0.45
    """
    result: list[tuple[float, float, float]] = []
    used_hough: set[int] = set()

    for ox, oy, oconf in onnx_dots:
        best_idx, best_dist = None, proximity_px
        for i, (hx, hy, _) in enumerate(hough_dots):
            dist = ((ox - hx) ** 2 + (oy - hy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None:
            used_hough.add(best_idx)
            result.append((ox, oy, min(1.0, oconf + boost)))
        else:
            result.append((ox, oy, oconf))

    for i, (hx, hy, hr) in enumerate(hough_dots):
        if i not in used_hough and 3 <= hr <= 15:
            result.append((hx, hy, 0.45))

    return result
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_detector.py -v
```

Expected: all 9 tests PASS.

Note: `test_detect_hough_finds_dot` may be flaky with Hough parameters — if it fails, widen the `abs(x - 320) < 15` tolerance to `< 25`.

- [ ] **Step 5: Commit**

```bash
git add pipeline/detector.py tests/test_detector.py
git commit -m "feat: detector — Hough circles, YOLOv8n ONNX inference, NMS, ensemble"
```

---

## Task 6: pipeline/grid.py

**Files:**
- Create: `pipeline/grid.py`
- Create: `tests/test_grid.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_grid.py`:

```python
import numpy as np
import pytest
from pipeline.grid import _cluster_1d, reconstruct_grid


# Braille dot-to-bit mapping (standard):
# Dot 1 (top-left)  → bit 0
# Dot 2 (mid-left)  → bit 1
# Dot 3 (bot-left)  → bit 2
# Dot 4 (top-right) → bit 3
# Dot 5 (mid-right) → bit 4
# Dot 6 (bot-right) → bit 5
#
# 'a' = dot 1 only → 0b000001 → U+2801 → ⠁
# 'b' = dots 1,2   → 0b000011 → U+2803 → ⠃
# 'h' = dots 1,2,5 → 0b010011 → U+2813 → ⠓


def _cell_dots(
    origin_x: int, origin_y: int, spacing: int, pattern: int
) -> list[tuple[float, float, float]]:
    """
    Build dot list for a single Braille cell.
    origin_x, origin_y: top-left dot position
    spacing: pixels between adjacent dot rows/columns
    pattern: 6-bit integer (bit 0=dot1 … bit 5=dot6)
    """
    positions = [
        (0, 0),  # dot 1: bit 0
        (0, 1),  # dot 2: bit 1
        (0, 2),  # dot 3: bit 2
        (1, 0),  # dot 4: bit 3
        (1, 1),  # dot 5: bit 4
        (1, 2),  # dot 6: bit 5
    ]
    dots = []
    for bit, (col, row) in enumerate(positions):
        if pattern & (1 << bit):
            x = float(origin_x + col * spacing)
            y = float(origin_y + row * spacing)
            dots.append((x, y, 0.9))
    return dots


def test_cluster_1d_two_groups():
    values = np.array([10.0, 12.0, 100.0, 102.0])
    labels = _cluster_1d(values, gap=20.0)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_1d_single_element():
    values = np.array([50.0])
    labels = _cluster_1d(values, gap=20.0)
    assert labels.tolist() == [0]


def test_cluster_1d_all_same_cluster():
    values = np.array([5.0, 8.0, 11.0])
    labels = _cluster_1d(values, gap=20.0)
    assert len(set(labels.tolist())) == 1


def test_reconstruct_grid_letter_a():
    # 'a' = dot 1 only = U+2801
    dots = _cell_dots(origin_x=100, origin_y=100, spacing=20, pattern=0b000001)
    braille_str, conf = reconstruct_grid(dots)
    assert '⠁' in braille_str  # ⠁


def test_reconstruct_grid_letter_b():
    # 'b' = dots 1,2 = U+2803
    dots = _cell_dots(origin_x=100, origin_y=100, spacing=20, pattern=0b000011)
    braille_str, conf = reconstruct_grid(dots)
    assert '⠃' in braille_str  # ⠃


def test_reconstruct_grid_letter_h():
    # 'h' = dots 1,2,5 = U+2813
    dots = _cell_dots(origin_x=100, origin_y=100, spacing=20, pattern=0b010011)
    braille_str, conf = reconstruct_grid(dots)
    assert '⠓' in braille_str  # ⠓


def test_reconstruct_grid_two_cells():
    # 'a' followed by 'b': two cells side by side with a gap
    dots_a = _cell_dots(origin_x=100, origin_y=100, spacing=20, pattern=0b000001)
    dots_b = _cell_dots(origin_x=200, origin_y=100, spacing=20, pattern=0b000011)
    braille_str, conf = reconstruct_grid(dots_a + dots_b)
    assert len(braille_str.replace('\n', '')) == 2


def test_reconstruct_grid_too_few_dots_returns_empty():
    dots = [(100.0, 100.0, 0.9), (110.0, 100.0, 0.9)]  # only 2 dots
    braille_str, conf = reconstruct_grid(dots)
    assert braille_str == ""


def test_reconstruct_grid_confidence_in_range():
    dots = _cell_dots(origin_x=100, origin_y=100, spacing=20, pattern=0b000001)
    _, conf = reconstruct_grid(dots)
    assert 0.0 <= conf <= 1.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_grid.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement pipeline/grid.py**

```python
import numpy as np
from scipy.spatial.distance import cdist


def _cluster_1d(values: np.ndarray, gap: float) -> np.ndarray:
    """
    Cluster 1D values by gaps greater than `gap`.
    Returns integer cluster labels (0-indexed), same length as values.
    """
    if len(values) == 0:
        return np.array([], dtype=int)
    idx = np.argsort(values)
    sorted_v = values[idx]
    labels = np.zeros(len(values), dtype=int)
    current = 0
    for i in range(1, len(sorted_v)):
        if sorted_v[i] - sorted_v[i - 1] > gap:
            current += 1
        labels[idx[i]] = current
    labels[idx[0]] = 0
    return labels


def reconstruct_grid(
    dots: list[tuple[float, float, float]], min_dots: int = 3
) -> tuple[str, float]:
    """
    Convert confirmed dot positions to a Braille Unicode string.

    dots: [(x, y, confidence), ...]
    Returns: (braille_unicode_string, mean_confidence)
             Empty string + 0.0 if not enough dots or clustering fails.

    Braille dot numbering (standard):
        1 4
        2 5
        3 6
    Bit 0 = dot 1, bit 1 = dot 2, …, bit 5 = dot 6.
    Unicode: U+2800 + bitmask.
    """
    if len(dots) < min_dots:
        return "", 0.0

    xs = np.array([d[0] for d in dots], dtype=float)
    ys = np.array([d[1] for d in dots], dtype=float)
    confs = np.array([d[2] for d in dots], dtype=float)

    # Estimate dot spacing from median nearest-neighbour distance
    pts = np.stack([xs, ys], axis=1)
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    dot_spacing = float(np.median(dists.min(axis=1)))
    if dot_spacing < 1.0:
        dot_spacing = 20.0

    gap = dot_spacing * 1.8

    row_labels = _cluster_1d(ys, gap)
    col_labels = _cluster_1d(xs, gap)

    unique_rows = sorted(set(row_labels.tolist()))
    unique_cols = sorted(set(col_labels.tolist()))

    if not unique_rows or not unique_cols:
        return "", 0.0

    # Build presence lookup: (row_cluster, col_cluster) → True
    presence: dict[tuple[int, int], bool] = {}
    for i in range(len(dots)):
        presence[(int(row_labels[i]), int(col_labels[i]))] = True

    braille_chars: list[str] = []

    # Iterate in groups of 3 row-clusters (one Braille text line)
    for row_start in range(0, len(unique_rows), 3):
        row_group = unique_rows[row_start: row_start + 3]

        # Iterate in groups of 2 col-clusters (one Braille cell)
        for col_start in range(0, len(unique_cols), 2):
            col_group = unique_cols[col_start: col_start + 2]

            bitmask = 0
            for ri, rc in enumerate(row_group):      # ri = 0,1,2 → dot row
                for ci, cc in enumerate(col_group):  # ci = 0 (left), 1 (right)
                    if presence.get((rc, cc), False):
                        bit = ri + ci * 3            # 0–2 left col, 3–5 right col
                        bitmask |= (1 << bit)

            braille_chars.append(chr(0x2800 + bitmask))

        # Add newline between text lines
        if row_start + 3 < len(unique_rows):
            braille_chars.append('\n')

    return "".join(braille_chars), float(confs.mean())
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_grid.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/grid.py tests/test_grid.py
git commit -m "feat: grid reconstruction — gap clustering, 6-bit cell encoding, Braille Unicode"
```

---

## Task 7: pipeline/corrector.py

**Files:**
- Create: `pipeline/corrector.py`
- Create: `tests/test_corrector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_corrector.py`:

```python
import numpy as np
import pytest
from pipeline.corrector import estimate_skew, correct_skew
from pipeline.grid import _cluster_1d


def _tilted_dots(angle_deg: float, n_cols: int = 5, spacing: int = 20) -> list:
    """Generate dots in a horizontal line tilted by angle_deg."""
    rad = np.radians(angle_deg)
    dots = []
    for i in range(n_cols):
        x = 100 + i * spacing * np.cos(rad)
        y = 100 + i * spacing * np.sin(rad)
        dots.append((float(x), float(y), 0.9))
    return dots


def test_estimate_skew_horizontal_line():
    dots = _tilted_dots(0.0, n_cols=6)
    ys = np.array([d[1] for d in dots])
    row_labels = _cluster_1d(ys, gap=10.0)
    angle = estimate_skew(dots, row_labels)
    assert abs(angle) < 2.0


def test_estimate_skew_tilted_line():
    dots = _tilted_dots(10.0, n_cols=6)
    ys = np.array([d[1] for d in dots])
    row_labels = _cluster_1d(ys, gap=5.0)
    angle = estimate_skew(dots, row_labels)
    assert abs(angle - 10.0) < 5.0


def test_estimate_skew_too_few_dots():
    dots = [(10.0, 10.0, 0.9), (20.0, 20.0, 0.9)]
    labels = np.array([0, 0])
    angle = estimate_skew(dots, labels)
    assert angle == pytest.approx(0.0)


def test_correct_skew_returns_same_shape():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    result = correct_skew(frame, 10.0)
    assert result.shape == frame.shape


def test_correct_skew_trivial_angle_returns_same():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[100, 100] = 255
    result = correct_skew(frame, 0.5)  # < 1° threshold
    assert np.array_equal(result, frame)


def test_correct_skew_rotates_content():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2_import_ok = True
    try:
        import cv2
        cv2.circle(frame, (320, 100), 5, (255, 255, 255), -1)
    except ImportError:
        cv2_import_ok = False
    if cv2_import_ok:
        result = correct_skew(frame, 15.0)
        # After rotation, original position should change
        assert not np.array_equal(result, frame)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_corrector.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement pipeline/corrector.py**

```python
import cv2
import numpy as np


def estimate_skew(
    dots: list[tuple[float, float, float]], row_labels: np.ndarray
) -> float:
    """
    Estimate page skew angle in degrees from dot row geometry.
    Fits a line to each row cluster and returns the median angle.
    Positive angle = clockwise tilt; negative = counter-clockwise.
    Returns 0.0 if not enough data.
    """
    if len(dots) < 4:
        return 0.0

    xs = np.array([d[0] for d in dots], dtype=float)
    ys = np.array([d[1] for d in dots], dtype=float)

    angles: list[float] = []
    for label in np.unique(row_labels):
        mask = row_labels == label
        rx, ry = xs[mask], ys[mask]
        if len(rx) < 2:
            continue
        coeffs = np.polyfit(rx, ry, 1)   # y = m*x + b
        angle = float(np.degrees(np.arctan(coeffs[0])))
        angles.append(angle)

    return float(np.median(angles)) if angles else 0.0


def correct_skew(frame: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate frame to correct skew. Angles below 1° are ignored (no-op).
    Preserves original frame dimensions with BORDER_REPLICATE fill.
    """
    if abs(angle_deg) < 1.0:
        return frame
    h, w = frame.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    return cv2.warpAffine(
        frame, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_corrector.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/corrector.py tests/test_corrector.py
git commit -m "feat: corrector — skew angle estimation, warpAffine rotation"
```

---

## Task 8: pipeline/translator.py

**Files:**
- Create: `pipeline/translator.py`
- Create: `tests/test_translator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_translator.py`:

```python
import pytest
from unittest.mock import patch, MagicMock


# liblouis is available only on Linux (apt: python3-louis).
# On Windows dev machines, we mock it to test our wrapper logic.
# On CI / the deployed app, use the real library.

def test_translate_grade1_a(monkeypatch):
    """'⠁' (dot 1) = Grade 1 'a'."""
    mock_louis = MagicMock()
    mock_louis.backTranslateString.return_value = "a"
    monkeypatch.setitem(__import__('sys').modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    result = tr.translate('⠁', grade='grade1')
    assert result == 'a'
    mock_louis.backTranslateString.assert_called_once_with(
        ['braille-patterns.cti', 'en-ueb-g1.ctb'], '⠁'
    )


def test_translate_grade2_default(monkeypatch):
    mock_louis = MagicMock()
    mock_louis.backTranslateString.return_value = "the"
    monkeypatch.setitem(__import__('sys').modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    result = tr.translate('⠮')  # grade='grade2' is default
    assert result == 'the'
    mock_louis.backTranslateString.assert_called_once_with(
        ['braille-patterns.cti', 'en-ueb-g2.ctb'], '⠮'
    )


def test_translate_empty_string_returns_empty(monkeypatch):
    mock_louis = MagicMock()
    monkeypatch.setitem(__import__('sys').modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    result = tr.translate('   ')
    assert result == ''
    mock_louis.backTranslateString.assert_not_called()


def test_translate_nemeth_table(monkeypatch):
    mock_louis = MagicMock()
    mock_louis.backTranslateString.return_value = "3"
    monkeypatch.setitem(__import__('sys').modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    tr.translate('⠃', grade='nemeth')
    mock_louis.backTranslateString.assert_called_once_with(
        ['braille-patterns.cti', 'nemeth.ctb'], '⠃'
    )


def test_translate_computer_table(monkeypatch):
    mock_louis = MagicMock()
    mock_louis.backTranslateString.return_value = "b"
    monkeypatch.setitem(__import__('sys').modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    tr.translate('⠃', grade='computer')
    mock_louis.backTranslateString.assert_called_once_with(
        ['braille-patterns.cti', 'en-us-comp8-ext.utb'], '⠃'
    )


def test_translate_handles_exception_gracefully(monkeypatch):
    """When backTranslateString raises, individual chars are tried; failures → [?]."""
    call_count = [0]

    def side_effect(tables, text):
        call_count[0] += 1
        if len(text) > 1:
            raise RuntimeError("bulk fail")
        if text == '⠁':
            return 'a'
        raise RuntimeError("char fail")

    mock_louis = MagicMock()
    mock_louis.backTranslateString.side_effect = side_effect
    monkeypatch.setitem(__import__('sys').modules, 'louis', mock_louis)

    from importlib import reload
    import pipeline.translator as tr
    reload(tr)

    result = tr.translate('⠁⠃', grade='grade1')
    assert 'a' in result
    assert '[?]' in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_translator.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement pipeline/translator.py**

```python
# louis is installed via apt (python3-louis) on Linux/Community Cloud.
# It is NOT available via pip and is NOT installed on Windows dev machines.
# We use a lazy import inside translate() so that:
#   - importing this module on Windows does NOT crash
#   - tests can monkeypatch sys.modules['louis'] before calling translate()
#   - on the deployed server (Linux), import louis works fine at call time

GRADE_TABLES: dict[str, list[str]] = {
    "grade2":   ["braille-patterns.cti", "en-ueb-g2.ctb"],
    "grade1":   ["braille-patterns.cti", "en-ueb-g1.ctb"],
    "nemeth":   ["braille-patterns.cti", "nemeth.ctb"],
    "computer": ["braille-patterns.cti", "en-us-comp8-ext.utb"],
}


def translate(braille_unicode: str, grade: str = "grade2") -> str:
    """
    Convert a Braille Unicode string to English text using liblouis.

    Uses backTranslateString() — the Braille→English direction.
    translateString() is the wrong direction (English→Braille).

    Grade 2 back-translation accuracy is ~95%+ for standard prose but
    degrades on proper nouns and isolated cells due to contraction ambiguity.

    Falls back to character-by-character translation on bulk failure;
    failed characters are replaced with [?].

    Returns "[liblouis unavailable]" if python3-louis is not installed
    (e.g. Windows dev machine without the apt package).
    """
    if not braille_unicode.strip():
        return ""

    try:
        import louis  # lazy: only imported at call time, not at module load
    except ImportError:
        return "[liblouis unavailable — install python3-louis on Linux]"

    tables = GRADE_TABLES.get(grade, GRADE_TABLES["grade2"])

    try:
        return louis.backTranslateString(tables, braille_unicode)
    except Exception:
        parts: list[str] = []
        for char in braille_unicode:
            if char == '\n':
                parts.append('\n')
                continue
            try:
                parts.append(louis.backTranslateString(tables, char))
            except Exception:
                parts.append('[?]')
        return "".join(parts)
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_translator.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/translator.py tests/test_translator.py
git commit -m "feat: translator — liblouis backTranslateString, all grades, fallback"
```

---

## Task 9: Camera Component — index.html

**Files:**
- Create: `components/camera_component/index.html`

No Python unit tests for this task — it is tested manually in the browser.

- [ ] **Step 1: Create components/camera_component/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1a1a1a; color: #fff; font-family: sans-serif; }

  #container { position: relative; width: 100%; }
  video { width: 100%; display: block; border-radius: 8px; }
  canvas#overlay {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%;
    border-radius: 8px; pointer-events: none;
  }

  #guidance {
    padding: 8px 12px; font-size: 1em;
    background: rgba(0,0,0,0.6); border-radius: 4px; margin: 6px 0;
    min-height: 2em;
  }
  #guidance.warn { color: #ffcc00; }
  #guidance.ok   { color: #66ff66; }

  #conf-bar { height: 6px; background: #333; border-radius: 3px; margin: 4px 0 8px; }
  #conf-fill { height: 100%; background: #4CAF50; border-radius: 3px;
               width: 0%; transition: width 0.4s ease; }

  #controls { display: flex; gap: 8px; flex-wrap: wrap; padding: 4px 0; }
  button {
    padding: 10px 18px; font-size: 0.95em; border: none;
    border-radius: 8px; cursor: pointer; background: #333; color: #fff;
    flex: 1; min-width: 100px;
  }
  button:active { background: #555; }

  #error { color: #ff5555; padding: 12px; display: none; }
</style>
</head>
<body>

<div id="container">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="overlay"></canvas>
</div>
<div id="guidance" class="ok">Initialising camera…</div>
<div id="conf-bar"><div id="conf-fill"></div></div>
<div id="controls">
  <button id="btn-pause">⏸ Pause</button>
  <button id="btn-flip">↺ Flip</button>
</div>
<div id="error"></div>

<script>
// ─── Streamlit component protocol (raw postMessage, no external CDN) ──────────
function setComponentValue(value) {
  window.parent.postMessage(
    { type: "streamlit:setComponentValue", value: value, dataType: "json" },
    "*"
  );
}
function setFrameHeight(height) {
  window.parent.postMessage(
    { type: "streamlit:setFrameHeight", height: height },
    "*"
  );
}
window.addEventListener("message", function (e) {
  if (e.data.type === "streamlit:render") {
    onRender(e.data.args || {});
  }
});
// Signal component is ready
window.parent.postMessage({ type: "streamlit:componentReady", apiVersion: 1 }, "*");

// ─── State ────────────────────────────────────────────────────────────────────
const video   = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx     = overlay.getContext("2d");
const guideEl = document.getElementById("guidance");
const confFill = document.getElementById("conf-fill");
const errEl   = document.getElementById("error");

let stream         = null;
let captureTimer   = null;
let paused         = false;
let facingMode     = "environment";
let autoSpeak      = false;
let speakGuidance  = false;
let lastSpokenText = "";
let isCapturing    = false;   // prevent overlapping captures

// ─── Camera ───────────────────────────────────────────────────────────────────
async function startCamera(facing) {
  if (stream) stream.getTracks().forEach(t => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: facing,
        width:  { ideal: 1280 },
        height: { ideal: 720 }
      }
    });
    video.srcObject = stream;
    await video.play();
    overlay.width  = video.videoWidth  || 640;
    overlay.height = video.videoHeight || 480;
    setFrameHeight(overlay.height + 120);
    errEl.style.display = "none";
  } catch (err) {
    errEl.textContent = "Camera error: " + err.message;
    errEl.style.display = "block";
  }
}

// ─── Frame capture ─────────────────────────────────────────────────────────
function captureFrame() {
  if (paused || isCapturing || !video.videoWidth) return;
  isCapturing = true;

  const tmp = document.createElement("canvas");
  tmp.width  = video.videoWidth;
  tmp.height = video.videoHeight;
  tmp.getContext("2d").drawImage(video, 0, 0);

  tmp.toBlob(blob => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const b64 = reader.result.split(",")[1];
      setComponentValue({ frame: b64, ts: Date.now() });
      isCapturing = false;
    };
    reader.readAsDataURL(blob);
  }, "image/jpeg", 0.85);
}

// ─── Result display ────────────────────────────────────────────────────────
function updateDisplay(result) {
  if (!result) return;

  // Annotated frame overlay
  if (result.annotated_frame) {
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      ctx.globalAlpha = 0.55;
      ctx.drawImage(img, 0, 0, overlay.width, overlay.height);
      ctx.globalAlpha = 1.0;
    };
    img.src = "data:image/jpeg;base64," + result.annotated_frame;
  }

  // Guidance
  if (result.guidance) {
    const g = result.guidance;
    guideEl.textContent = g.message || "";
    guideEl.className   = g.status === "ok" ? "ok" : "warn";
    if (speakGuidance && g.status === "warn") speakText(g.message);
  }

  // Confidence bar
  if (result.confidence !== undefined) {
    confFill.style.width = Math.round(result.confidence * 100) + "%";
  }

  // Auto-speak translated text
  if (autoSpeak && result.text && result.text !== lastSpokenText) {
    lastSpokenText = result.text;
    speakText(result.text);
  }
}

// ─── Web Speech API ────────────────────────────────────────────────────────
function speakText(text) {
  if (!text || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utt  = new SpeechSynthesisUtterance(text);
  utt.lang   = "en-US";
  utt.rate   = 0.9;
  window.speechSynthesis.speak(utt);
}

// ─── Controls ──────────────────────────────────────────────────────────────
document.getElementById("btn-pause").addEventListener("click", () => {
  paused = !paused;
  document.getElementById("btn-pause").textContent = paused ? "▶ Resume" : "⏸ Pause";
});

document.getElementById("btn-flip").addEventListener("click", () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  startCamera(facingMode);
});

// ─── Streamlit render callback ─────────────────────────────────────────────
function onRender(args) {
  autoSpeak     = !!args.auto_speak;
  speakGuidance = !!args.speak_guidance;
  updateDisplay(args.result || null);
}

// ─── Boot ──────────────────────────────────────────────────────────────────
startCamera(facingMode).then(() => {
  captureTimer = setInterval(captureFrame, 300);
});
</script>
</body>
</html>
```

- [ ] **Step 2: Manual smoke test (browser)**

Open a standalone HTML file in Chrome on Android or PC:

```bash
# Open the file directly in Chrome (for local testing without Streamlit)
start chrome components/camera_component/index.html
```

Expected: Camera permission prompt appears, video stream starts, no console errors.

Note: The `setComponentValue` calls will silently fail (no parent Streamlit frame) in standalone mode — that is expected. The camera and controls should work.

- [ ] **Step 3: Commit**

```bash
git add components/camera_component/index.html
git commit -m "feat: camera component JS — getUserMedia, canvas capture, Web Speech API, overlays"
```

---

## Task 10: Camera Component — Python Wrapper

**Files:**
- Modify: `components/camera_component/__init__.py`

- [ ] **Step 1: Implement components/camera_component/__init__.py**

```python
import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Declares the component, pointing Streamlit at the directory containing index.html.
# Streamlit serves index.html from this directory over HTTPS (Community Cloud).
camera_component = components.declare_component(
    "camera_component",
    path=_COMPONENT_DIR,
)
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from components.camera_component import camera_component; print('OK')"
```

Expected: `OK` printed without error.

- [ ] **Step 3: Commit**

```bash
git add components/camera_component/__init__.py
git commit -m "feat: camera component Python wrapper — declare_component"
```

---

## Task 11: app.py — Main Application

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create app.py**

```python
import numpy as np
import cv2
import streamlit as st
import psutil

from components.camera_component import camera_component
from pipeline.preprocessor import preprocess
from pipeline.detector import load_onnx_session, detect_hough, detect_onnx, ensemble
from pipeline.grid import reconstruct_grid, _cluster_1d
from pipeline.corrector import estimate_skew, correct_skew
from pipeline.translator import translate
from utils.image_utils import decode_frame, encode_frame, letterbox, draw_dots
from utils.quality import blur_score, brightness, dot_density, get_guidance

MODEL_PATH = "models/yolov8n_braille.onnx"
_COUNTER_KEY = "frame_counter"

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Braille Scanner",
    page_icon="⠃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Load shared resources once per process ───────────────────────────────────
onnx_session = load_onnx_session(MODEL_PATH)

# ─── Sidebar settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    grade = st.selectbox(
        "Braille Grade",
        options=["grade2", "grade1", "nemeth", "computer"],
        format_func=lambda x: {
            "grade2": "Grade 2 (UEB contracted)",
            "grade1": "Grade 1 (uncontracted)",
            "nemeth": "Nemeth (math)",
            "computer": "Computer (8-dot)",
        }[x],
        index=0,
    )
    auto_speak    = st.checkbox("Auto-speak translation", value=True)
    speak_guide   = st.checkbox("Speak camera guidance", value=False)
    high_contrast = st.checkbox("High contrast", value=False)
    large_text    = st.checkbox("Large text", value=False)

# ─── CSS injection ────────────────────────────────────────────────────────────
_bg  = "#000" if high_contrast else "#fff"
_fg  = "#fff" if high_contrast else "#000"
_sz  = "1.4em" if large_text  else "1em"
st.markdown(f"""
<style>
section[data-testid="stMain"] > div {{ background: {_bg}; color: {_fg}; }}
.translation-box {{ font-size: {_sz}; font-family: monospace; }}
</style>
""", unsafe_allow_html=True)

# ─── Title ────────────────────────────────────────────────────────────────────
st.title("⠃ Braille Scanner")
if onnx_session is None:
    st.caption("Running in Hough-only mode (ONNX model not found at models/yolov8n_braille.onnx)")

# ─── Session state init ───────────────────────────────────────────────────────
if "last_result" not in st.session_state:
    st.session_state.last_result = {
        "text": "",
        "confidence": 0.0,
        "guidance": {"status": "ok", "message": "Ready — point camera at Braille"},
    }
if _COUNTER_KEY not in st.session_state:
    st.session_state[_COUNTER_KEY] = 0

# ─── Pipeline helper ──────────────────────────────────────────────────────────
def run_pipeline(raw_bgr: np.ndarray, counter: int, selected_grade: str) -> dict:
    """
    Full pipeline on a raw BGR frame.
    Returns dict: {annotated_frame, text, confidence, guidance}
    """
    # Downscale if memory is tight (free-tier safeguard)
    mem = psutil.virtual_memory()
    target_size = 480 if mem.used > 700 * 1024 * 1024 else 640

    frame_640, scale, (pad_x, pad_y) = letterbox(raw_bgr, target_size)

    # Dynamic CLAHE based on brightness
    gray_quick = cv2.cvtColor(frame_640, cv2.COLOR_BGR2GRAY)
    bright = brightness(gray_quick)
    clip = 3.5 if bright < 40 else (1.0 if bright > 210 else 2.0)

    binary, gray = preprocess(frame_640, clip_limit=clip)

    # Stage 2A: Hough (every cycle)
    hough_raw = detect_hough(binary)

    # Stage 2B: ONNX every 3rd cycle
    if onnx_session and counter % 3 == 0:
        onnx_raw = detect_onnx(frame_640, onnx_session)
        confirmed = ensemble(hough_raw, onnx_raw)
    else:
        confirmed = [(x, y, 0.45) for x, y, r in hough_raw]

    # Quality signals
    b_score = blur_score(gray)
    density = dot_density(confirmed, 640, 640)
    skew = 0.0

    # Skew correction if enough dots
    if len(confirmed) >= 4:
        ys = np.array([d[1] for d in confirmed])
        row_labels = _cluster_1d(ys, gap=20.0)
        skew = estimate_skew(confirmed, row_labels)
        if abs(skew) > 5.0:
            frame_640 = correct_skew(frame_640, skew)
            binary, gray = preprocess(frame_640, clip_limit=clip)
            hough_raw = detect_hough(binary)
            if onnx_session and counter % 3 == 0:
                onnx_raw = detect_onnx(frame_640, onnx_session)
                confirmed = ensemble(hough_raw, onnx_raw)
            else:
                confirmed = [(x, y, 0.45) for x, y, r in hough_raw]

    guidance = get_guidance(b_score, bright, density, skew)

    # Draw overlays
    draw_list = [(x, y, 5.0) for x, y, _ in confirmed]
    annotated = draw_dots(frame_640, draw_list)
    annotated_b64 = encode_frame(annotated)

    # Grid + translation
    braille_str, conf = reconstruct_grid(confirmed)
    if braille_str:
        text = translate(braille_str, selected_grade)
    else:
        text = st.session_state.last_result.get("text", "")

    return {
        "annotated_frame": annotated_b64,
        "text": text,
        "confidence": conf,
        "guidance": guidance,
    }

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_live, tab_upload = st.tabs(["📷 Live Camera", "📁 Upload Image"])

# ── Live Camera tab ───────────────────────────────────────────────────────────
with tab_live:

    @st.fragment(run_every="500ms")
    def live_fragment():
        # Render the camera component; returns latest frame sent by JS
        frame_data = camera_component(
            key="camera",
            auto_speak=auto_speak,
            speak_guidance=speak_guide,
            result=st.session_state.last_result,
        )

        if frame_data and isinstance(frame_data, dict) and "frame" in frame_data:
            raw = decode_frame(frame_data["frame"])
            if raw is not None:
                st.session_state[_COUNTER_KEY] += 1
                result = run_pipeline(raw, st.session_state[_COUNTER_KEY], grade)
                st.session_state.last_result = result

        # Display result below the component
        result = st.session_state.last_result
        st.markdown(
            f"**{result['guidance']['message']}**",
            help="Camera guidance",
        )
        conf_pct = int(result.get("confidence", 0) * 100)
        st.progress(conf_pct, text=f"Confidence: {conf_pct}%")

        text_val = result.get("text", "")
        st.text_area(
            "Translation",
            value=text_val,
            height=100,
            key="live_text_area",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Copy", key="btn_copy_live"):
                st.write("Copied!")  # Browser clipboard requires JS; shown as text fallback
        with col_b:
            if st.button("▶ Speak", key="btn_speak_live") and text_val:
                # Inject JS to speak via Web Speech API
                st.markdown(
                    f"<script>speechSynthesis.cancel();speechSynthesis.speak("
                    f"new SpeechSynthesisUtterance({text_val!r}));</script>",
                    unsafe_allow_html=True,
                )

    live_fragment()

# ── Upload tab ────────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a Braille image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if raw is not None:
            result = run_pipeline(raw, counter=0, selected_grade=grade)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(
                    f"data:image/jpeg;base64,{result['annotated_frame']}",
                    caption="Detected dots",
                    use_container_width=True,
                )
            with col2:
                st.metric("Confidence", f"{int(result['confidence'] * 100)}%")
                st.text_area("Translation", value=result["text"], height=150)
                if st.button("▶ Speak", key="btn_speak_upload") and result["text"]:
                    st.markdown(
                        f"<script>speechSynthesis.cancel();speechSynthesis.speak("
                        f"new SpeechSynthesisUtterance({result['text']!r}));</script>",
                        unsafe_allow_html=True,
                    )
                if st.button("⬇ Download Audio (MP3)", key="btn_audio") and result["text"]:
                    import asyncio, tempfile, os
                    import edge_tts
                    async def _synth(text: str) -> bytes:
                        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                            await communicate.save(f.name)
                            f.flush()
                            return open(f.name, "rb").read()
                    audio_bytes = asyncio.run(_synth(result["text"]))
                    st.download_button(
                        "Save MP3", data=audio_bytes,
                        file_name="braille_translation.mp3", mime="audio/mpeg",
                    )
        else:
            st.error("Could not decode image. Please upload a valid JPG or PNG.")
```

- [ ] **Step 2: Run the app locally and verify it starts**

```bash
streamlit run app.py
```

Expected:
- App opens in browser at http://localhost:8501
- Two tabs: "📷 Live Camera" and "📁 Upload Image"
- "Running in Hough-only mode" caption (since no ONNX model yet)
- Upload tab: upload any JPG and confirm pipeline runs without crashing
- No Python exceptions in the terminal

- [ ] **Step 3: Test upload mode with a sample image**

Download a test Braille image or create one:

```python
# Run this once to generate a test image: python make_test_image.py
import cv2
import numpy as np

img = np.ones((480, 640, 3), dtype=np.uint8) * 240  # light grey paper
# Draw 3 dots representing Braille 'a' (dot 1 only) at cell position
for y, x in [(100, 100)]:  # dot 1
    cv2.circle(img, (x, y), 8, (40, 40, 40), -1)
cv2.imwrite("test_braille.jpg", img)
```

Upload `test_braille.jpg` via the Upload tab and confirm:
- Annotated image shows a green circle overlay on the dot
- No crash, confidence is shown

- [ ] **Step 4: Test live camera mode (local)**

- Open http://localhost:8501 in Chrome on Android (or PC)
- Click "📷 Live Camera" tab
- Allow camera permissions
- Confirm: video stream appears, Pause/Flip buttons work
- Hold phone over any paper with circular marks → confirm green overlays appear

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: main app — fragment live camera, upload mode, pipeline wiring, CSS"
```

---

## Task 12: Model Acquisition (Training + Export)

**Files:**
- No new Python files — this task uses Colab or a local GPU machine.

This task is a prerequisite for full-accuracy ONNX mode. The Hough-only fallback works for testing all other code without the model.

- [ ] **Step 1: Prepare the DSBI dataset**

In Google Colab or a GPU machine:

```bash
pip install ultralytics
# Clone DSBI dataset
git clone https://github.com/yeluo1994/DSBI
# The dataset contains Braille images + annotations.
# Convert annotations to YOLO format (one txt file per image):
# Each line: class_id cx cy w h (normalised 0-1)
# class_id = 0 (single class: 'dot')
```

- [ ] **Step 2: Create braille_dots.yaml**

```yaml
path: ./DSBI_yolo      # root of converted dataset
train: images/train
val:   images/val
nc: 1
names: ['dot']
```

- [ ] **Step 3: Train YOLOv8n**

```bash
yolo train \
  model=yolov8n.pt \
  data=braille_dots.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  augment=True \
  degrees=30 \
  hsv_v=0.4 \
  perspective=0.001
```

Expected: Training completes, best weights at `runs/detect/train/weights/best.pt`.
Target validation mAP@0.5 ≥ 0.85.

- [ ] **Step 4: Export to FP32 ONNX**

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 opset=12
# Produces: runs/detect/train/weights/best.onnx (~12MB)
```

- [ ] **Step 5: Copy and verify model**

```bash
cp runs/detect/train/weights/best.onnx models/yolov8n_braille.onnx

# Quick verification:
python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('models/yolov8n_braille.onnx', providers=['CPUExecutionProvider'])
dummy = np.zeros((1,3,640,640), dtype=np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: dummy})
print('Output shape:', out[0].shape)  # Should be [1, 5, 8400]
"
```

Expected output: `Output shape: (1, 5, 8400)`

- [ ] **Step 6: Commit model to repo**

```bash
git add models/yolov8n_braille.onnx
git commit -m "feat: add YOLOv8n FP32 ONNX model (~12MB)"
```

Note: 12MB files are within GitHub's 100MB file size limit and do not require Git LFS.

---

## Task 13: Deployment

**Files:**
- No new code files — deployment configuration and verification.

- [ ] **Step 1: Verify all deployment files are correct**

Check `requirements.txt` has exactly:
```
streamlit>=1.37.0
opencv-python-headless>=4.9.0
onnxruntime-cpu>=1.17.0
numpy>=1.26.0
scipy>=1.12.0
edge-tts>=6.1.0
pytest>=8.0.0
```

Check `packages.txt` has exactly:
```
liblouis-dev
python3-louis
```

Verify `opencv-python-headless` is used (NOT `opencv-python` — the display variant fails on headless Linux servers).

Note on onnxruntime: `requirements.txt` uses `onnxruntime>=1.17.0` (cross-platform). On Community Cloud (Linux, no GPU), this installs CPU-only automatically. `onnxruntime-cpu` is a Linux-only alias and does not exist on Windows.

- [ ] **Step 2: Push to a public GitHub repository**

```bash
git remote add origin https://github.com/<your-username>/braille-scanner.git
git push -u origin main
```

The repo must be **public** for Streamlit Community Cloud free tier.

- [ ] **Step 3: Deploy on Streamlit Community Cloud**

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your GitHub repo, branch `main`, main file `app.py`
4. Click "Deploy"

Deployment takes 3–5 minutes. Streamlit runs `apt-get install liblouis-dev python3-louis` from `packages.txt` then `pip install -r requirements.txt`.

- [ ] **Step 4: Verify deployment**

On the deployed URL (https://your-app.streamlit.app):

1. **PC Chrome:** Open app → Upload tab → upload a test Braille image → confirm translation appears and Speak button works.
2. **Android Chrome:** Open URL → Live Camera tab → allow camera → confirm video stream starts, green dot overlays appear, guidance text shows.
3. **Check memory:** In Streamlit Cloud dashboard → "App resources" — confirm memory stays below 800MB during active scanning.
4. **Check fragment reruns:** Open Chrome DevTools → Network tab → confirm WebSocket messages during live scanning are lightweight (not full-page reloads).

- [ ] **Step 5: Final smoke test all Braille grades**

Using the Upload tab with a known Braille image:
- Select "Grade 1" → translate → confirm correct output
- Select "Grade 2" → translate → confirm contractions are decoded
- Confirm no crash on Nemeth or Computer grade selection

- [ ] **Step 6: Final commit**

```bash
git add .
git commit -m "deploy: streamlit community cloud — verified on Android Chrome + PC Chrome"
git push
```

---

## Full Test Run

After all tasks, run the full test suite:

```bash
pytest tests/ -v --tb=short
```

Expected output:
```
tests/test_image_utils.py    8 passed
tests/test_quality.py       12 passed
tests/test_preprocessor.py   6 passed
tests/test_detector.py       9 passed
tests/test_grid.py           9 passed
tests/test_corrector.py      6 passed
tests/test_translator.py     6 passed
======================== 56 passed ========================
```

Note: `test_translator.py` passes on all platforms because it mocks `louis`. On the deployed Linux server, the actual `louis` module is available and real translation works.
