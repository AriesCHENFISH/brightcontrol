from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

MODEL_ID = "/data1/cx/sd15"
PROMPT_FILE = Path("prompts/pilot_prompts_day1.txt")
OUT_ROOT = Path("results/day1_vanilla")
IMG_DIR = OUT_ROOT / "images"
CSV_PATH = OUT_ROOT / "pilot_metrics.csv"
HIST_PATH = OUT_ROOT / "pilot_brightness_hist.png"
GRID_PATH = OUT_ROOT / "case_grid.png"

NUM_STEPS = 30
GUIDANCE_SCALE = 7.5
HEIGHT = 512
WIDTH = 512
SEEDS = [0, 1, 2, 3]

def compute_mean_luminance(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(y.mean())

def make_case_grid(image_paths, save_path, cols=4, thumb_size=(256, 256)):
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = ImageOps.fit(img, thumb_size)
        images.append(img)

    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), "white")

    for i, img in enumerate(images):
        x = (i % cols) * thumb_size[0]
        y = (i // cols) * thumb_size[1]
        grid.paste(img, (x, y))

    grid.save(save_path)

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    prompts = [x.strip() for x in PROMPT_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not prompts:
        raise ValueError("No prompts found.")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()

    rows = []
    saved_for_grid = []

    for prompt_idx, prompt in enumerate(prompts):
        for seed in SEEDS:
            generator = torch.Generator(device=device).manual_seed(seed)
            result = pipe(
                prompt=prompt,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                height=HEIGHT,
                width=WIDTH,
            )
            image = result.images[0]
            mean_lum = compute_mean_luminance(image)

            filename = f"p{prompt_idx:02d}_s{seed}.png"
            image_path = IMG_DIR / filename
            image.save(image_path)

            if seed == 0:
                saved_for_grid.append(str(image_path))

            rows.append({
                "prompt_id": prompt_idx,
                "seed": seed,
                "prompt": prompt,
                "image_path": str(image_path),
                "mean_luminance": mean_lum,
            })
            print(f"prompt={prompt_idx:02d} seed={seed} lum={mean_lum:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.hist(df["mean_luminance"], bins=20, edgecolor="black")
    plt.xlabel("Mean luminance")
    plt.ylabel("Image count")
    plt.title("SD1.5 vanilla: mean luminance histogram")
    plt.tight_layout()
    plt.savefig(HIST_PATH, dpi=200)
    plt.close()

    make_case_grid(saved_for_grid[:16], GRID_PATH, cols=4)

    print(df["mean_luminance"].describe())
    print(f"Saved: {CSV_PATH}")
    print(f"Saved: {HIST_PATH}")
    print(f"Saved: {GRID_PATH}")

if __name__ == "__main__":
    main()