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
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 opset=12
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
