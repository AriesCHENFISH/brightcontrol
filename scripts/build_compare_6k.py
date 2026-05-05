from pathlib import Path
import json
import shutil
import random
import csv

ROOT = Path("/data1/cx/brightcontrol")
COCO_DIR = ROOT / "data/raw/coco"
IMG_DIR = COCO_DIR / "train2017"
STATS_CSV = COCO_DIR / "coco_train2017_stats.csv"

OUT_DIR = ROOT / "data/train_compare_6k/imagefolder"
OUT_IMG_DIR = OUT_DIR
META_PATH = OUT_DIR / "metadata.jsonl"

RANDOM_SEED = 42

N_DARK = 2000
N_BRIGHT = 2000
N_NEUTRAL = 2000

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
                "image_path": IMG_DIR / row["file_name"]
            })
    return rows

def main():
    random.seed(RANDOM_SEED)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

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

    final = []
    for x in dark_sel:
        x = dict(x)
        x["brightness_label"] = "dark"
        x["text"] = add_brightness_prefix(x["caption"], "dark")
        final.append(x)

    for x in bright_sel:
        x = dict(x)
        x["brightness_label"] = "bright"
        x["text"] = add_brightness_prefix(x["caption"], "bright")
        final.append(x)

    for x in neutral_sel:
        x = dict(x)
        x["brightness_label"] = "neutral"
        x["text"] = add_brightness_prefix(x["caption"], "neutral")
        final.append(x)

    random.shuffle(final)

    with open(META_PATH, "w", encoding="utf-8") as f:
        for x in final:
            dst = OUT_IMG_DIR / x["file_name"]
            if not dst.exists():
                if x["image_path"].exists():
                    shutil.copy2(x["image_path"], dst)
                else:
                    print(f"Warning: source image not found: {x['image_path']}")
                    continue

            item = {
                "file_name": x["file_name"],
                "text": x["text"]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"built {len(final)} samples")
    print(f"dark: {len(dark_sel)}, bright: {len(bright_sel)}, neutral: {len(neutral_sel)}")
    print(f"metadata: {META_PATH}")
    print(f"images dir: {OUT_IMG_DIR}")

if __name__ == "__main__":
    main()