from pathlib import Path
import json
import shutil
import random

import numpy as np
from PIL import Image

ROOT = Path("/data1/cx/brightcontrol")
COCO_DIR = ROOT / "data/raw/coco"
IMG_DIR = COCO_DIR / "val2017"
CAPTION_JSON = COCO_DIR / "annotations/captions_val2017.json"

OUT_DIR = ROOT / "data/train_smoke/imagefolder"
OUT_IMG_DIR = OUT_DIR
META_PATH = OUT_DIR / "metadata.jsonl"

RANDOM_SEED = 42

N_DARK = 250
N_BRIGHT = 250
N_NEUTRAL = 500

def mean_luminance(path: Path) -> float:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(y.mean())

def choose_longest_caption(captions):
    captions = [c.strip() for c in captions if c.strip()]
    return max(captions, key=len)

def add_brightness_prefix(caption, label):
    if label == "dark":
        return f"a very dark photo of {caption}"
    elif label == "bright":
        return f"a very bright photo of {caption}"
    else:
        return f"a normally lit photo of {caption}"

def main():
    random.seed(RANDOM_SEED)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CAPTION_JSON, "r", encoding="utf-8") as f:
        ann = json.load(f)

    id_to_file = {x["id"]: x["file_name"] for x in ann["images"]}
    caps = {}
    for x in ann["annotations"]:
        caps.setdefault(x["image_id"], []).append(x["caption"])

    rows = []
    image_ids = sorted(caps.keys())
    for image_id in image_ids:
        file_name = id_to_file[image_id]
        img_path = IMG_DIR / file_name
        if not img_path.exists():
            continue

        lum = mean_luminance(img_path)
        caption = choose_longest_caption(caps[image_id])

        rows.append({
            "image_id": image_id,
            "file_name": file_name,
            "image_path": img_path,
            "mean_luminance": lum,
            "caption": caption,
        })

    rows = sorted(rows, key=lambda x: x["mean_luminance"])

    n = len(rows)
    dark_pool = rows[: int(0.2 * n)]
    bright_pool = rows[int(0.8 * n):]
    neutral_pool = rows[int(0.4 * n): int(0.6 * n)]

    dark_sel = random.sample(dark_pool, N_DARK)
    bright_sel = random.sample(bright_pool, N_BRIGHT)
    neutral_sel = random.sample(neutral_pool, N_NEUTRAL)

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
                shutil.copy2(x["image_path"], dst)

            item = {
                "file_name": x["file_name"],
                "text": x["text"]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"built {len(final)} samples")
    print(f"metadata: {META_PATH}")
    print(f"images dir: {OUT_IMG_DIR}")

if __name__ == "__main__":
    main()