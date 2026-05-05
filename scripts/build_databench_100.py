from pathlib import Path
import json
import random
import csv

ROOT = Path("/data1/cx/brightcontrol")
COCO_DIR = ROOT / "data/raw/coco"
IMG_DIR = COCO_DIR / "val2017"
STATS_CSV = COCO_DIR / "coco_val2017_stats.csv"

OUT_JSON = ROOT / "data/databench_100.json"

RANDOM_SEED = 42

N_DARK = 40
N_BRIGHT = 40
N_NEUTRAL = 20

def add_brightness_prefix(caption, label):
    if label == "dark":
        return f"a very dark photo of {caption}"
    elif label == "bright":
        return f"a very bright photo of {caption}"
    else:
        return f"a normally lit photo of {caption}"

def load_stats(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image_id": int(row["image_id"]),
                "file_name": row["file_name"],
                "mean_luminance": float(row["mean_luminance"]),
                "caption": row["caption"],
                "image_path": str(IMG_DIR / row["file_name"])
            })
    return rows

def main():
    random.seed(RANDOM_SEED)

    rows = load_stats(STATS_CSV)
    rows = sorted(rows, key=lambda x: x["mean_luminance"])

    n = len(rows)
    dark_pool = rows[: int(0.2 * n)]
    bright_pool = rows[int(0.8 * n):]
    neutral_pool = rows[int(0.4 * n): int(0.6 * n)]

    if len(dark_pool) < N_DARK:
        print(f"Warning: dark pool size {len(dark_pool)} < required {N_DARK}")
        N_DARK_ACTUAL = len(dark_pool)
    else:
        N_DARK_ACTUAL = N_DARK

    if len(bright_pool) < N_BRIGHT:
        print(f"Warning: bright pool size {len(bright_pool)} < required {N_BRIGHT}")
        N_BRIGHT_ACTUAL = len(bright_pool)
    else:
        N_BRIGHT_ACTUAL = N_BRIGHT

    if len(neutral_pool) < N_NEUTRAL:
        print(f"Warning: neutral pool size {len(neutral_pool)} < required {N_NEUTRAL}")
        N_NEUTRAL_ACTUAL = len(neutral_pool)
    else:
        N_NEUTRAL_ACTUAL = N_NEUTRAL

    dark_sel = random.sample(dark_pool, N_DARK_ACTUAL)
    bright_sel = random.sample(bright_pool, N_BRIGHT_ACTUAL)
    neutral_sel = random.sample(neutral_pool, N_NEUTRAL_ACTUAL)

    databench = []
    for x in dark_sel:
        item = {
            "image_id": x["image_id"],
            "file_name": x["file_name"],
            "image_path": x["image_path"],
            "mean_luminance": x["mean_luminance"],
            "caption": x["caption"],
            "brightness_label": "dark",
            "text": add_brightness_prefix(x["caption"], "dark")
        }
        databench.append(item)

    for x in bright_sel:
        item = {
            "image_id": x["image_id"],
            "file_name": x["file_name"],
            "image_path": x["image_path"],
            "mean_luminance": x["mean_luminance"],
            "caption": x["caption"],
            "brightness_label": "bright",
            "text": add_brightness_prefix(x["caption"], "bright")
        }
        databench.append(item)

    for x in neutral_sel:
        item = {
            "image_id": x["image_id"],
            "file_name": x["file_name"],
            "image_path": x["image_path"],
            "mean_luminance": x["mean_luminance"],
            "caption": x["caption"],
            "brightness_label": "neutral",
            "text": add_brightness_prefix(x["caption"], "neutral")
        }
        databench.append(item)

    random.shuffle(databench)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(databench, f, indent=2, ensure_ascii=False)

    print(f"built DataBench-100 with {len(databench)} samples")
    print(f"dark: {len(dark_sel)}, bright: {len(bright_sel)}, neutral: {len(neutral_sel)}")
    print(f"saved to {OUT_JSON}")

if __name__ == "__main__":
    main()