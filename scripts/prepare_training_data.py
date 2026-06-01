"""
Prepare YOLO training data from Angelina and DSBI braille datasets.

Run from repo root after cloning both datasets alongside each other:
  git clone https://github.com/IlyaOvodov/AngelinaDataset AngelinaDataset
  git clone https://github.com/yeluo1994/DSBI DSBI

Usage:
  python scripts/prepare_training_data.py \
      --angelina /path/to/AngelinaDataset \
      --dsbi /path/to/DSBI \
      --out data/braille_dots

Output directory layout expected by braille_dots.yaml:
  data/braille_dots/images/{train,val}/
  data/braille_dots/labels/{train,val}/

Issue resolved during training (2025-05):
- Angelina JSONs use labelme rectangle format; x1/y1 may not be top-left —
  coordinates are normalised with min/max so negative boxes are clipped.
- DSBI .txt files encode cell boundaries as x-coords on line 1, y-coords on
  line 2, then cell rows starting line 3 with (row, col, dot1..dot6) tuples.
  A "dsbi_" prefix is added to avoid filename collisions with Angelina images.
"""

import argparse
import glob
import json
import os
import shutil

import cv2
from sklearn.model_selection import train_test_split


def _makedirs(base: str) -> None:
    for split in ("train", "val"):
        os.makedirs(os.path.join(base, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base, "labels", split), exist_ok=True)


# ── Angelina dataset ─────────────────────────────────────────────────────────

def _load_angelina(root: str) -> list[tuple[str, str, dict]]:
    records = []
    for j_path in glob.glob(os.path.join(root, "**/*.json"), recursive=True):
        if "not_braille" in j_path:
            continue
        img_path = j_path.replace(".labeled.json", ".labeled.jpg")
        if not os.path.exists(img_path):
            img_path = j_path.replace(".json", ".jpg")
        if not os.path.exists(img_path):
            continue
        try:
            with open(j_path) as f:
                data = json.load(f)
            records.append((j_path, img_path, data))
        except json.JSONDecodeError:
            continue
    return records


def _write_angelina(records, base: str, split: str) -> int:
    count = 0
    for _, img_path, data in records:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            continue
        base_name = os.path.basename(img_path).rsplit(".", 1)[0]
        label_path = os.path.join(base, "labels", split, f"{base_name}.txt")
        new_img_path = os.path.join(base, "images", split, f"{base_name}.jpg")
        with open(label_path, "w") as lf:
            for shape in data.get("shapes", []):
                if shape.get("shape_type") != "rectangle":
                    continue
                pts = shape["points"]
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                cx = max(0.0, min(1.0, ((xmin + xmax) / 2.0) / w))
                cy = max(0.0, min(1.0, ((ymin + ymax) / 2.0) / h))
                bw = max(0.0, min(1.0, (xmax - xmin) / w))
                bh = max(0.0, min(1.0, (ymax - ymin) / h))
                lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        shutil.copy(img_path, new_img_path)
        count += 1
    return count


# ── DSBI dataset ─────────────────────────────────────────────────────────────

def _load_dsbi(root: str) -> list[tuple[str, str]]:
    records = []
    for txt_path in glob.glob(os.path.join(root, "**/*.txt"), recursive=True):
        img_path = txt_path.replace(".txt", ".jpg")
        if os.path.exists(img_path):
            records.append((txt_path, img_path))
    return records


def _write_dsbi(records, base: str, split: str) -> int:
    count = 0
    for txt_path, img_path in records:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            continue
        # "dsbi_" prefix prevents filename collision with Angelina images
        base_name = "dsbi_" + os.path.basename(img_path).rsplit(".", 1)[0]
        label_path = os.path.join(base, "labels", split, f"{base_name}.txt")
        new_img_path = os.path.join(base, "images", split, f"{base_name}.jpg")
        with open(txt_path) as f:
            lines = f.readlines()
        if len(lines) < 3:
            continue
        # Line 0: scale factor (unused for bbox conversion)
        # Line 1: x column boundary coordinates
        # Line 2: y row boundary coordinates
        # Line 3+: (row, col, dot1..dot6) cell entries
        x_coords = list(map(int, lines[1].strip().split()))
        y_coords = list(map(int, lines[2].strip().split()))
        with open(label_path, "w") as lf:
            for line in lines[3:]:
                parts = list(map(int, line.strip().split()))
                if len(parts) < 8:
                    continue
                r, c = parts[0], parts[1]
                try:
                    x1 = x_coords[c]
                    x2 = x_coords[c + 1] if c + 1 < len(x_coords) else x1 + 20
                    y1 = y_coords[r]
                    y2 = y_coords[r + 1] if r + 1 < len(y_coords) else y1 + 30
                    cx = max(0.0, min(1.0, ((x1 + x2) / 2.0) / w))
                    cy = max(0.0, min(1.0, ((y1 + y2) / 2.0) / h))
                    bw = max(0.0, min(1.0, (x2 - x1) / w))
                    bh = max(0.0, min(1.0, (y2 - y1) / h))
                    lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                except IndexError:
                    pass
        shutil.copy(img_path, new_img_path)
        count += 1
    return count


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angelina", required=True, help="Path to AngelinaDataset root")
    parser.add_argument("--dsbi", required=True, help="Path to DSBI root")
    parser.add_argument("--out", default="data/braille_dots", help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _makedirs(args.out)

    # Angelina
    angelina_records = _load_angelina(args.angelina)
    train_a, val_a = train_test_split(angelina_records, test_size=args.val_ratio, random_state=args.seed)
    n_a_train = _write_angelina(train_a, args.out, "train")
    n_a_val = _write_angelina(val_a, args.out, "val")
    print(f"Angelina: {n_a_train} train, {n_a_val} val")

    # DSBI
    dsbi_records = _load_dsbi(args.dsbi)
    train_d, val_d = train_test_split(dsbi_records, test_size=args.val_ratio, random_state=args.seed)
    n_d_train = _write_dsbi(train_d, args.out, "train")
    n_d_val = _write_dsbi(val_d, args.out, "val")
    print(f"DSBI:     {n_d_train} train, {n_d_val} val")

    print(f"Total:    {n_a_train + n_d_train} train, {n_a_val + n_d_val} val")
    print(f"Output:   {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
