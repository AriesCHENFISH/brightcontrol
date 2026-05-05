from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

MODEL_ID = "/data1/cx/sd15"
PROMPT_FILE = Path("prompts/pilot_prompts.txt")
OUT_ROOT = Path("results/pilot_run")
CSV_PATH = OUT_ROOT / "pilot_metrics.csv"
FIG_PATH = OUT_ROOT / "pilot_brightness_hist.png"

NUM_STEPS = 30
GUIDANCE_SCALE = 7.5
HEIGHT = 512
WIDTH = 512
SEEDS = [0, 1, 2, 3]

SETTINGS = {
    "vanilla_ddim": {
        "rescale_betas_zero_snr": False,
        "timestep_spacing": "leading",
    },
    "zero_snr_ddim": {
        "rescale_betas_zero_snr": True,
        "timestep_spacing": "trailing",
    },
}


def compute_mean_luminance(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(y.mean())


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    prompts = [x.strip() for x in PROMPT_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(prompts) == 0:
        raise ValueError("No prompts found.")

    print(f"Loading pipeline from {MODEL_ID} ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    # Optional speed / memory optimization
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers enabled.")
    except Exception as e:
        print(f"xformers not enabled: {e}")
        pipe.enable_attention_slicing()

    rows = []

    for setting_name, sched_kwargs in SETTINGS.items():
        print(f"\n=== Running setting: {setting_name} ===")
        setting_dir = OUT_ROOT / setting_name
        setting_dir.mkdir(parents=True, exist_ok=True)

        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            **sched_kwargs
        )

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
                image_path = setting_dir / filename
                image.save(image_path)

                rows.append({
                    "setting": setting_name,
                    "prompt_id": prompt_idx,
                    "seed": seed,
                    "prompt": prompt,
                    "image_path": str(image_path),
                    "mean_luminance": mean_lum,
                })

                print(f"[{setting_name}] prompt={prompt_idx:02d} seed={seed} mean_lum={mean_lum:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    plt.figure(figsize=(9, 5))
    for setting_name, subdf in df.groupby("setting"):
        plt.hist(
            subdf["mean_luminance"],
            bins=20,
            alpha=0.55,
            label=setting_name,
            edgecolor="black",
        )
    plt.xlabel("Mean luminance")
    plt.ylabel("Image count")
    plt.title("Pilot prompts: mean luminance histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200)
    plt.close()

    summary = df.groupby("setting")["mean_luminance"].agg(["mean", "std", "min", "max", "count"])
    print("\n=== Summary ===")
    print(summary)
    print(f"\nSaved CSV to: {CSV_PATH}")
    print(f"Saved figure to: {FIG_PATH}")


if __name__ == "__main__":
    main()
