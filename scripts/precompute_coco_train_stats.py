from pathlib import Path
import json
import csv
import numpy as np
from PIL import Image
import argparse
import sys

ROOT = Path("/data1/cx/brightcontrol")
COCO_DIR = ROOT / "data/raw/coco"

def get_paths(dataset="train2017"):
    if dataset not in ("train2017", "val2017"):
        raise ValueError(f"dataset must be 'train2017' or 'val2017', got {dataset}")
    IMG_DIR = COCO_DIR / dataset
    CAPTION_JSON = COCO_DIR / f"annotations/captions_{dataset}.json"
    OUT_CSV = ROOT / f"data/raw/coco/coco_{dataset}_stats.csv"
    return IMG_DIR, CAPTION_JSON, OUT_CSV

def mean_luminance(path: Path) -> float:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(y.mean())

def choose_longest_caption(captions):
    captions = [c.strip() for c in captions if c.strip()]
    return max(captions, key=len) if captions else ""

def compute_stats(img_dir, caption_json, out_csv):
    with open(caption_json, "r", encoding="utf-8") as f:
        ann = json.load(f)

    id_to_file = {x["id"]: x["file_name"] for x in ann["images"]}
    caps = {}
    for x in ann["annotations"]:
        caps.setdefault(x["image_id"], []).append(x["caption"])

    rows = []
    image_ids = sorted(caps.keys())

    for idx, image_id in enumerate(image_ids, start=1):
        file_name = id_to_file[image_id]
        img_path = img_dir / file_name
        if not img_path.exists():
            continue

        lum = mean_luminance(img_path)
        caption = choose_longest_caption(caps[image_id])

        rows.append({
            "image_id": image_id,
            "file_name": file_name,
            "mean_luminance": lum,
            "caption": caption,
        })

        if idx % 2000 == 0:
            print(f"processed {idx} images")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "file_name", "mean_luminance", "caption"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved to {out_csv}")
    print(f"num rows: {len(rows)}")

def main():
    parser = argparse.ArgumentParser(description="Compute luminance stats for COCO dataset.")
    parser.add_argument("--dataset", type=str, default="train2017",
                        choices=["train2017", "val2017"],
                        help="Dataset split to process (train2017 or val2017)")
    args = parser.parse_args()

    img_dir, caption_json, out_csv = get_paths(args.dataset)

    if not img_dir.exists():
        print(f"Error: Image directory {img_dir} does not exist.", file=sys.stderr)
        print("Please download/extract the COCO images.", file=sys.stderr)
        sys.exit(1)

    if not caption_json.exists():
        print(f"Error: Caption JSON file {caption_json} does not exist.", file=sys.stderr)
        sys.exit(1)

    compute_stats(img_dir, caption_json, out_csv)

if __name__ == "__main__":
    main()