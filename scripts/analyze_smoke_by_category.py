import pandas as pd
from pathlib import Path

IN_PATH = Path("results/day2_compare/by_prompt_summary.csv")
OUT_PATH = Path("results/day2_compare/by_category_summary.csv")

PROMPT_CATEGORY = {
    0: "dark",
    1: "bright",
    2: "white_bg",
    3: "black_bg",
    4: "backlight",
    5: "dark",
    6: "neutral",
    7: "neutral",
}

df = pd.read_csv(IN_PATH)
df["category"] = df["prompt_id"].map(PROMPT_CATEGORY)

# Dynamically identify mean_lum and delta_vs_base columns
mean_lum_cols = [col for col in df.columns if col.endswith("_mean_lum")]
delta_cols = [col for col in df.columns if col.endswith("_delta_vs_base")]
agg_cols = mean_lum_cols + delta_cols

# Group by category and compute means
out = df.groupby("category", as_index=False)[agg_cols].mean()
out.to_csv(OUT_PATH, index=False, encoding="utf-8")

print("Category Summary:")
print(out)
print(f"\nSaved to {OUT_PATH}")