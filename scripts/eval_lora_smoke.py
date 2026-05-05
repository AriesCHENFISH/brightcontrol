from pathlib import Path
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

def compute_mean_luminance(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(y.mean())

def make_grid(image_paths, save_path, cols=4, thumb_size=(256, 256)):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    prompts = [x.strip() for x in Path(args.prompt_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    seeds = [0, 1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
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

    if args.lora_dir:
        pipe.load_lora_weights(args.lora_dir)

    rows = []
    grid_paths = []

    for prompt_idx, prompt in enumerate(prompts):
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
                height=args.height,
                width=args.width,
            ).images[0]

            lum = compute_mean_luminance(image)
            img_path = img_dir / f"p{prompt_idx:02d}_s{seed}.png"
            image.save(img_path)

            if seed == 0:
                grid_paths.append(str(img_path))

            rows.append({
                "prompt_id": prompt_idx,
                "seed": seed,
                "prompt": prompt,
                "mean_luminance": lum,
                "image_path": str(img_path),
            })
            print(f"prompt={prompt_idx:02d} seed={seed} lum={lum:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.hist(df["mean_luminance"], bins=16, edgecolor="black")
    plt.xlabel("Mean luminance")
    plt.ylabel("Image count")
    plt.title("Lora smoke eval: mean luminance histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "hist.png", dpi=200)
    plt.close()

    make_grid(grid_paths, out_dir / "grid.png", cols=4)

    print(df["mean_luminance"].describe())
    print(f"saved to {out_dir}")

if __name__ == "__main__":
    main()