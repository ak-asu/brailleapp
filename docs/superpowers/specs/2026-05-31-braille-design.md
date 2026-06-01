# Braille Scanner — Design Spec
**Date:** 2026-05-31  
**Status:** Approved (v2 — post-verification)  
**Target:** Streamlit Community Cloud (free tier) · Android Chrome · PC Chrome/Edge/Firefox

---

## Problem Statement

Build a system that uses a camera to scan real physical Braille (embossed or handwritten) and convert it into English text and speech in real-time or near real-time. Must work on phone browsers (Android Chrome) and PC browsers without paid infrastructure.

---

## Verified Constraints & Decisions

| Constraint | Decision | Verification Source |
|---|---|---|
| Free tier ~1GB RAM, CPU only | `onnxruntime-cpu` + no torch | ONNX-only stack ~450–600MB total |
| No paid services | Custom JS component, not streamlit-webrtc | streamlit-webrtc requires paid TURN on Community Cloud; documented broken 2024–2025 |
| Full rerun per setComponentValue call | Wrap processing in `@st.fragment(run_every="500ms")` | Fragment-scoped reruns confirmed stable in Streamlit 1.37 (July 2024) |
| Android + PC (no iOS) | `getUserMedia({facingMode:"environment"})` | Works natively on Android Chrome + PC browsers over HTTPS |
| liblouis Python binding | `python3-louis` via `packages.txt` (apt), NOT pip | `pip install louis` is unreliable; apt package is the stable path |
| Braille → English direction | `louis.backTranslateString()` | `translateString()` is English→Braille (wrong direction) |
| Grade default | Grade 2 (UEB) + manual override | Auto-detection from cell patterns is unsound before table selection |
| ONNX model format | FP32 ONNX primary; INT8 optional post-export step | `int8=True` in ultralytics ONNX export has known issues; static quantization via `onnxruntime.quantization` is the reliable INT8 path |

### WebGPU — Why Rejected (Documented)
Client-side ONNX Runtime Web + WebGPU was explored and would give true 10–30 FPS. Rejected for this prototype because: (1) requires a custom JS Streamlit component with full ONNX Runtime Web integration (~400+ lines of JS), (2) liblouis back-translation still needs server-side Python, (3) WebGPU on Android requires Chrome 113+ / Android 12+ — an unnecessary constraint for this build. **If the @st.fragment approach proves too slow in production, client-side WebGPU inference is the documented pivot path.**

---

## Project Structure

```
braille/
├── app.py                         # Streamlit entrypoint
├── requirements.txt
├── packages.txt
├── models/
│   └── yolov8n_braille.onnx       # FP32 ONNX, pre-exported, committed to repo (~12MB)
├── components/
│   └── camera_component/
│       ├── __init__.py            # Streamlit component wrapper
│       └── index.html             # JS: getUserMedia, canvas capture, Web Speech API, overlays
├── pipeline/
│   ├── preprocessor.py            # CLAHE, blur, adaptive threshold
│   ├── detector.py                # Hough + YOLOv8n ONNX ensemble
│   ├── grid.py                    # DBSCAN clustering, cell construction
│   ├── corrector.py               # Perspective/homography correction
│   └── translator.py              # liblouis all-grade Braille → English
└── utils/
    ├── image_utils.py             # encode/decode base64, frame drawing
    └── quality.py                 # blur score, brightness, alignment signals
```

---

## Tech Stack

**Python dependencies (`requirements.txt`):**
```
streamlit>=1.37.0
opencv-python-headless>=4.9.0
onnxruntime-cpu>=1.17.0
numpy>=1.26.0
scipy>=1.12.0
edge-tts>=6.1.0
```

**System dependencies (`packages.txt`):**
```
liblouis-dev
python3-louis
```

Note: `liblouis-dev` provides the C library; `python3-louis` provides the `import louis` Python binding. Do NOT add `louis` to `requirements.txt` — there is no reliable pip package.

**Estimated memory footprint:**
- Streamlit baseline: ~150MB
- opencv-python-headless: ~80MB
- onnxruntime-cpu: ~60MB
- numpy + scipy: ~40MB
- YOLOv8n FP32 ONNX model in memory: ~30MB
- python3-louis + liblouis: ~10MB
- Overhead/buffers: ~80MB
- **Total: ~450–600MB — within 1GB free tier limit with ~400MB headroom**

---

## Architecture Overview

```
BROWSER (Android Chrome / PC Chrome)
├── Custom JS Component (index.html)
│   ├── getUserMedia({facingMode:"environment"}) → 30fps live video (local, zero server)
│   ├── Canvas frame capture every 300ms → JPEG (quality=0.85) → base64
│   ├── Streamlit.setComponentValue({frame, ts}) → triggers fragment rerun
│   ├── Receives result {annotated_frame, text, guidance, confidence} from Python
│   ├── Overlays annotated frame on live video display
│   ├── Shows guidance text + confidence bar
│   ├── Web Speech API → speaks translated text (zero server latency)
│   └── Controls: front/rear toggle, pause/resume, torch toggle (if supported)
│
└── Streamlit App (app.py)
    ├── Main script: renders static UI, grade selector, settings
    │
    ├── @st.fragment(run_every="500ms")  ← scoped rerun, NOT full app rerun
    │   ├── Reads latest frame from component value (st.session_state)
    │   ├── Runs pipeline (preprocessor → detector → grid → translator)
    │   ├── Writes result back to component via st.session_state
    │   └── Renders: annotated frame, translated text, confidence, guidance
    │
    └── Tab 2: st.file_uploader() → same pipeline, single-shot, no fragment needed

PYTHON SERVER (Streamlit Community Cloud)
├── @st.cache_resource: ONNX InferenceSession (loaded once, shared across sessions)
└── @st.cache_resource: liblouis table path (loaded once)
```

### How @st.fragment Solves the Rerun Problem
Without fragments, every `Streamlit.setComponentValue()` call from JS triggers a **full app rerun**, making 2–3 Hz video processing unsustainable on the free tier. With `@st.fragment(run_every="500ms")`, only the fragment's code reruns — the rest of the app (sidebar, settings, static UI) stays frozen. This is the standard 2025 Streamlit pattern for streaming/real-time data and was stabilised in v1.37.

### Frame Backpressure (No Queue Needed)
If a fragment rerun arrives while the previous frame is still processing, Python simply returns the cached result from `st.session_state.last_result`. There is no explicit queue — the fragment always processes the newest available frame in session_state and discards older ones. This is safe because the JS component always overwrites `st.session_state.latest_frame` with the newest capture.

---

## Processing Pipeline

### Stage 1 — Preprocessing (every frame, ~5ms)
```
Raw JPEG (base64 decoded)
  → cv2.imdecode → resize to 640×640 (letterbox, preserve aspect, pad black)
  → cv2.cvtColor(BGR→GRAY)
  → cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply()   ← uneven lighting
  → cv2.GaussianBlur((5,5), 0)                                    ← noise reduction
  → cv2.adaptiveThreshold(Gaussian, blockSize=11, C=2)            ← any paper color
  → cv2.morphologyEx(MORPH_OPEN, kernel=3×3)                      ← noise speckles
```

### Stage 2A — Fast Path: Hough Circle Detection (every frame, ~10–15ms)
```
Preprocessed frame
  → cv2.HoughCircles(HOUGH_GRADIENT_ALT, dp=1.5, minDist=8,
                     param1=300, param2=0.85, minR=3, maxR=12)
  → dot list: [(x, y, r), ...]
  → draw overlays on original color frame (green circles)
  → store in session_state for ensemble use
```
Purpose: delivers live visual dot-position feedback at maximum speed with zero ML dependency.
Falls back to this path exclusively if ONNX session fails to load.

### Stage 2B — Accurate Path: YOLOv8n ONNX (every 3rd fragment cycle, ~60–80ms)
```
Preprocessed frame
  → Normalize to [0.0, 1.0], transpose to [1, 3, 640, 640] float32
  → onnxruntime InferenceSession.run(input_name, tensor)
  → Output: [1, 5, 8400] — (x_center, y_center, w, h, confidence) per anchor
  → NMS: confidence threshold=0.4, IoU threshold=0.45
  → Dot centers: [(x, y, confidence), ...]

Ensemble with Stage 2A Hough detections (same preprocessed frame):
  → For each ONNX detection: if Hough has a dot within 8px → confidence += 0.15
  → For each Hough detection with no ONNX match: include if radius is plausible (3–12px)
  → Final confirmed dot list: [(x, y, confidence), ...]
```
Rationale: Hough excels on clean, well-lit, uniform paper. YOLOv8n handles blur, low contrast,
embossed dots, worn Braille. Ensemble covers each other's failure modes.

### Stage 3 — Grid Reconstruction (~5ms)
```
Confirmed dot list
  → Quality gate: skip if len(dots) < 3 OR blur_score < 50
  → DBSCAN(eps=15, min_samples=1) on y-coords → row cluster labels
  → DBSCAN(eps=15, min_samples=1) on x-coords → column cluster labels
  → Median vertical spacing → cell_height (should be ~2× dot_spacing)
  → Median horizontal spacing → cell_width (should be ~1× dot_spacing)
  → For each cell (identified by row_pair × col_pair):
      → Snap dots to 2×3 positions: top/mid/bot × left/right
      → 6-bit binary pattern (dot 1–6 in standard Braille numbering)
      → Braille Unicode codepoint = U+2800 + bitmask
  → Ordered list of codepoints = Braille Unicode string
```

### Stage 4 — Perspective & Skew Correction (~3ms, conditional)
```
Triggered if: estimated row angle > 5° from horizontal

  → Extract row centers from DBSCAN row clusters
  → Fit line to each row's x,y centroids → skew angle θ
  → If |θ| > 5°: build rotation matrix, cv2.warpAffine()
  → If perspective detected (rows not parallel):
      → Use 4-point homography estimate from row/column extremes
      → cv2.warpPerspective() to rectify
  → Re-run Stage 3 on corrected frame

Handles: camera held at angle, curved book pages, slanted paper
```

### Stage 5 — Translation (~2ms)
```
Braille Unicode string
  → table selection (from user setting or default):
      Grade 2 (default): louis.backTranslateString(
                            ["braille-patterns.cti", "en-ueb-g2.ctb"],
                            braille_unicode_str)
      Grade 1:           louis.backTranslateString(
                            ["braille-patterns.cti", "en-ueb-g1.ctb"],
                            braille_unicode_str)
      Nemeth (math):     louis.backTranslateString(
                            ["braille-patterns.cti", "nemeth.ctb"],
                            braille_unicode_str)
      Computer (8-dot):  louis.backTranslateString(
                            ["braille-patterns.cti", "en-us-comp8-ext.utb"],
                            braille_unicode_str)
  → English text string
  → confidence: mean of per-cell confidence scores (0.0–1.0)

Note on Grade 2 back-translation accuracy:
  Grade 2 Braille uses 180+ contractions (e.g., ⠮ = "the", ⠯ = "and").
  Back-translation (Braille→English) via liblouis is inherently lossy for Grade 2
  due to contraction ambiguity. Accuracy is good (~95%+) for standard English prose
  but degrades for proper nouns, abbreviated content, and isolated cells.
  Grade 1 back-translation is lossless.
```

### Stage 6 — Quality & Guidance Signals (every 5th fragment cycle, ~3ms)
```
  blur_score   = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
  brightness   = gray_frame.mean()
  dot_density  = len(confirmed_dots) / (frame_w * frame_h) * 1e6  (dots/megapixel)
  skew_angle   = estimated from row geometry (Stage 4)

Guidance messages returned to JS (shown as overlay + optionally spoken):
  blur_score < 50     → "Hold camera steady"
  blur_score < 80     → "Almost — hold steadier"
  brightness < 40     → "Need more light"
  brightness > 210    → "Too bright — find shade"
  dot_density < 5     → "Move closer to the Braille"
  dot_density > 200   → "Move back slightly"
  abs(skew_angle) > 15 → "Tilt camera — align with page edge"
  all OK              → "Good — scanning..."
```

---

## UI Design

### Layout
```
┌─────────────────────────────────────────────────┐
│  Braille Scanner              [Live] [Upload]    │
├─────────────────────────────────────────────────┤
│  ┌──────────────── Camera Feed ───────────────┐  │
│  │  [Live video + green dot overlays]         │  │
│  │  "Good — scanning..."          [⏸ Pause]   │  │
│  │  Confidence: ████████░░ 82%   [↺ Flip cam] │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌──────────── Translated Text ───────────────┐  │
│  │  "Hello, my name is Sarah."               │  │
│  │                    [▶ Speak]  [⧉ Copy]     │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ▼ Settings (collapsed)                          │
│  Grade: [Grade 2 ▼]  Auto-speak: [✓]            │
│  High Contrast: [ ]  Large Text: [ ]             │
└─────────────────────────────────────────────────┘
```

### Modes
- **Live**: `@st.fragment` + custom JS component, auto-refresh at 500ms server cycle, 300ms JS capture
- **Upload**: `st.file_uploader()` (JPG/PNG), same full pipeline, single-shot, synchronous

### Accessibility
- High contrast toggle: CSS class swap (dark bg, white text, yellow highlights on Braille cells)
- Large text toggle: 1.4× font scale
- Auto-speak: every new translated text spoken via Web Speech API immediately
- Camera guidance optionally spoken aloud (toggle, off by default to avoid noise)
- Rear camera default on mobile (`facingMode: "environment"`)
- Front/rear camera switch button (re-calls getUserMedia with new constraint)
- Pause button: JS stops canvas capture interval

### Braille Grade Selector
- `Grade 2` — contracted UEB, 180+ rules, standard in books **(default)**
- `Grade 1` — direct letter mapping, lossless
- `Nemeth` — math notation
- `Computer` — 8-dot computer Braille (en-us-comp8-ext.utb)

No auto-detection. User selects grade. Grade 2 as default is correct because the overwhelming majority of physical Braille books use Grade 2.

---

## Error Handling & Robustness

### Image Quality Conditions

| Condition | Detection | Response |
|---|---|---|
| Motion blur | Laplacian variance < 80 | Hold last valid result, show "Hold steady" |
| Low light | Mean brightness < 40 | Boost CLAHE clipLimit to 3.5 dynamically |
| Overexposure | Mean brightness > 210 | Reduce CLAHE clipLimit to 1.0 |
| Camera too far | dot_density < 5 dots/MP | "Move closer" |
| Camera too close | Dots overlapping (radius > 0.5× spacing) | "Move back slightly" |
| Heavy skew > 25° | Row angle deviation | Auto-correct via homography |
| Double-sided bleed-through | Low-confidence ghost dots | Filter: confidence < 0.4 discarded |
| Partial frame at edges | Incomplete cells | Translate complete cells only; mark incomplete as [?] |

### Braille-Type-Specific Adaptations

| Braille Type | Adaptation |
|---|---|
| Handwritten / stylus-embossed | Widen Hough range: minR=2, maxR=15 |
| Old / worn (flattened dots) | Add morphological gradient before Hough |
| Plastic labels (glare) | `cv2.normalize()` + bilateral filter pre-CLAHE |
| Double-sided books | Confidence threshold enforced; area filter on small detections |
| Autofocus transition frames | Skip: Laplacian < 50, hold last valid result |

### Pipeline Failure Fallbacks
- ONNX session load fails → Hough-only path always active, no degradation in UX
- Grid reconstruction fails (< 3 dots or bad clusters) → return raw dot overlay, "No Braille detected — adjust position"
- liblouis back-translation fails for a cell → mark as [?], continue with remaining valid cells
- Fragment cycle arrives before processing completes → return `st.session_state.last_result` (cached), skip frame
- Memory approaches limit → detect via `psutil.virtual_memory()`, downscale input to 480×480

---

## Model Details

**YOLOv8n fine-tuned on Braille dot detection:**
- Base architecture: YOLOv8n (Ultralytics nano, 3.2M params)
- Training datasets: DSBI (114 double-sided images) + Angelina dataset (real smartphone photos) + augmented synthetic samples
- Augmentations: rotation ±30°, brightness ±40%, Gaussian noise σ=0.02, JPEG compression q=60–95, perspective warp ±10°
- Export format: **FP32 ONNX** (straightforward, reliable) — `yolo export model=best.pt format=onnx imgsz=640`
- File size: ~12MB FP32 ONNX (within git LFS or direct commit limit)
- Optional INT8: post-export static quantization via `onnxruntime.quantization.quantize_static()` with calibration data → ~6MB, ~1.5× speedup on CPU
- Input: `[1, 3, 640, 640]` float32, values in [0.0, 1.0]
- Output: `[1, 5, 8400]` — (x_center, y_center, width, height, confidence) per anchor
- No training happens at deploy time — model file committed to repo

**Model acquisition (one-time prerequisite, done locally or on Colab with GPU):**
```bash
# 1. Fine-tune YOLOv8n on Braille dot dataset
yolo train model=yolov8n.pt data=braille_dots.yaml epochs=100 imgsz=640

# 2. Export to ONNX FP32
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640

# 3. (Optional) INT8 static quantization
python quantize.py --input best.onnx --output yolov8n_braille_int8.onnx
# Uses onnxruntime.quantization.quantize_static() with ~50 calibration images

# 4. Commit to repo
git add models/yolov8n_braille.onnx
```

---

## Performance Profile (Streamlit Community Cloud Free Tier)

| Component | Time per frame |
|---|---|
| JS canvas capture + base64 encode | ~5ms (client, free) |
| Fragment rerun overhead | ~20–30ms |
| Preprocessing (CLAHE + threshold + morph) | ~5ms |
| Stage 2A: Hough detection | ~10–15ms |
| Stage 2B: YOLOv8n ONNX FP32 inference | ~60–80ms (every 3rd cycle) |
| Stage 3: DBSCAN grid reconstruction | ~5ms |
| Stage 4: Perspective correction (if needed) | ~3ms |
| Stage 5: liblouis back-translation | ~2ms |
| Network (Community Cloud → browser) | ~100–200ms |
| **Fragment cycle total (fast path)** | **~160–260ms** |
| **Fragment cycle total (accurate path)** | **~220–340ms** |
| **Effective annotated FPS** | **~2–3 FPS** |
| **Local video preview (JS, no server)** | **30 FPS** |

The user sees a smooth 30fps live video preview at all times. The annotated dot-overlay and translated text update at ~2–3 FPS from the server. This split gives the perception of real-time while staying within free-tier compute limits.

---

## Deployment Checklist

**Model prerequisite (one-time, done locally/Colab before first deploy):**
- [ ] Fine-tune YOLOv8n on DSBI + Angelina datasets
- [ ] Export FP32 ONNX: `yolo export model=best.pt format=onnx imgsz=640`
- [ ] (Optional) Run INT8 static quantization for speed
- [ ] Commit `models/yolov8n_braille.onnx` to repo

**Repository setup:**
- [ ] `packages.txt` contains `liblouis-dev` and `python3-louis` (one per line)
- [ ] `requirements.txt` uses `opencv-python-headless` (not `opencv-python` — no display system on server)
- [ ] `requirements.txt` uses `streamlit>=1.37.0` (fragments required)
- [ ] `louis` is NOT in `requirements.txt` (it comes from apt `python3-louis`)
- [ ] GitHub repo is public (Community Cloud requirement)
- [ ] ONNX model file ≤ 100MB (FP32 ~12MB — fine; use Git LFS only if > 100MB)

**Pre-submission testing:**
- [ ] Test on Android Chrome (rear camera, auto-speak, all grades)
- [ ] Test on PC Chrome (upload mode + live mode)
- [ ] Verify fragment reruns do not cause full app reload (check browser network tab)
- [ ] Verify liblouis `backTranslateString` works for Grade 1, 2, Nemeth, Computer
- [ ] Verify memory under 800MB during active scanning session
