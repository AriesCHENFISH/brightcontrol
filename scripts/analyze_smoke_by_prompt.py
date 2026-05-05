import pandas as pd
from pathlib import Path

RUNS = {
    "base": "results/lora_smoke_eval/base/metrics.csv",
    "baseline": "results/lora_smoke_eval/baseline_run1_final/metrics.csv",
    "vpred_200": "results/lora_smoke_eval/v_prediction_ckpt200/metrics.csv",
    "vpred_500": "results/lora_smoke_eval/v_prediction_ckpt500/metrics.csv",
    "offset002_200": "results/lora_smoke_eval/noise_offset_002_ckpt200/metrics.csv",
    "offset002_500": "results/lora_smoke_eval/noise_offset_002_ckpt500/metrics.csv",
    "offset005_200": "results/lora_smoke_eval/noise_offset_005_ckpt200/metrics.csv",
    "offset005_500": "results/lora_smoke_eval/noise_offset_005_ckpt500/metrics.csv",
}

OUT = Path("results/day2_compare/by_prompt_summary.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def load_mean_by_prompt(path):
    df = pd.read_csv(path)
    g = df.groupby(["prompt_id", "prompt"], as_index=False)["mean_luminance"].mean()
    return g.rename(columns={"mean_luminance": "mean_lum"})

tables = {}
for run_name, path in RUNS.items():
    if not Path(path).exists():
        print(f"Warning: {path} not found, skipping {run_name}")
        continue
    t = load_mean_by_prompt(path)
    t = t.rename(columns={"mean_lum": f"{run_name}_mean_lum"})
    tables[run_name] = t

if not tables:
    print("No data found!")
    exit(1)

merged = tables["base"]
for run_name in tables.keys():
    if run_name == "base":
        continue
    merged = merged.merge(tables[run_name], on=["prompt_id", "prompt"], how="left")

for run_name in tables.keys():
    if run_name == "base":
        continue
    merged[f"{run_name}_delta_vs_base"] = merged[f"{run_name}_mean_lum"] - merged["base_mean_lum"]

merged.to_csv(OUT, index=False, encoding="utf-8")
print(merged)
print(f"saved to {OUT}")