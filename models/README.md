# Models

## Available models

| File | Training data | Epochs | mAP50 (val) | Notes |
|------|--------------|--------|-------------|-------|
| `yolov8n_braille.onnx` | Angelina only | 100 | — | Baseline; single-dataset |
| `yolov8n_braille_combined.onnx` | Angelina + DSBI | 100 | — | **Primary model** — more training data, clearly superior (single-dataset model gives near-zero confidence on real images) |

`app.py` loads `yolov8n_braille_combined.onnx` by default.

## Architecture

YOLOv8n detector (nano), single class (`dot`).  
Input: `(1, 3, 640, 640)` float32 RGB.  
Output: `(1, 5, 8400)` — transposed to `(8400, 5)` as `[cx, cy, w, h, conf]` in pixel coordinates.  
Export: ONNX opset 12, FP32, ~12 MB.

## Confidence calibration

These models were trained on ~500 images total; confidence scores are lower than larger-corpus models.
The detection threshold in `pipeline/detector.py` is 0.25 (not the standard 0.4) to reflect this.
Hough circle detection runs every frame as primary detector; ONNX fires every 3rd frame as a secondary
validator via the ensemble function.

## Datasets

### AngelinaDataset
- Source: <https://github.com/IlyaOvodov/AngelinaDataset>
- Format: LabelMe JSON, rectangle annotations, one file per image
- Size used: 290 valid image-annotation pairs → 232 train / 58 val (80/20 split)
- Annotation format correction applied: `pts[0]` and `pts[1]` may not be top-left/bottom-right;
  `min/max` is applied and values are clipped to `[0, 1]` before writing YOLO labels.

### DSBI
- Source: <https://github.com/yeluo1994/DSBI>
- Format: custom `.txt` — line 0: scale, line 1: x-boundaries, line 2: y-boundaries,
  lines 3+: `(row, col, dot1..dot6)` cell entries
- Size used: 228 valid image-annotation pairs → 176 train / 44 val (80/20 split)
- Conversion: cell boundary indices mapped to YOLO bounding boxes; images prefixed with `dsbi_`
  to avoid filename collisions with Angelina images.

## Training

Training was performed on Google Colab (T4 GPU).

```
yolo train model=yolov8n.pt data=braille_dots.yaml \
    epochs=100 imgsz=640 batch=16 \
    degrees=30 hsv_v=0.4 perspective=0.001
```

Data augmentations:
- `degrees=30` — rotation up to ±30° (braille pages are often tilted)
- `hsv_v=0.4` — value jitter for lighting variation
- `perspective=0.001` — mild perspective warp

## Exporting

```
yolo export model=runs/detect/trainX/weights/best.pt format=onnx imgsz=640 opset=12
```

## Re-training

Use `scripts/prepare_training_data.py` to regenerate the dataset from raw sources:

```
python scripts/prepare_training_data.py \
    --angelina /path/to/AngelinaDataset \
    --dsbi /path/to/DSBI \
    --out data/braille_dots
```

Then re-run the training command above with the generated `data/braille_dots` directory.

## `.gitignore` note

`*.onnx` is globally ignored but `models/*.onnx` is explicitly allowed, so the model files
are tracked by git. Each is ~12 MB. Consider Git LFS if the repo grows large.
