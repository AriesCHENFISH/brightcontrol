import json
from pathlib import Path

OUT = Path("/data1/cx/brightcontrol/prompts/promptbench_v0_40.json")

groups = {
    "dark": [
        "a candle-lit dinner table in a very dark room",
        "an abandoned alley at midnight lit by a single street lamp",
        "a rainy city street at night with neon reflections, very dark scene",
        "a violinist playing on a dim stage with only one spotlight",
        "three hikers standing under an almost black night sky, silhouette",
        "a dark cave interior with a faint torch light",
        "a cat sitting in a dim kitchen at night",
        "a narrow street in heavy darkness with only distant lights",
    ],
    "bright": [
        "a cozy bedroom flooded with bright morning sunlight",
        "a snowy field under harsh midday sunlight, extremely bright",
        "an overexposed studio portrait on a seamless white background",
        "a beach at noon under intense sunlight, extremely bright",
        "a white marble hallway filled with strong daylight",
        "a city square under harsh summer noon light",
        "a bright greenhouse interior with sunlight everywhere",
        "a bright sunlit rooftop with washed-out highlights",
    ],
    "backlight": [
        "a person standing in front of a bright window, strong backlighting, silhouette",
        "tree branches against a glowing sunset sky, extreme backlighting",
        "a cyclist riding toward the sun, strong lens flare, silhouette",
        "a black cat standing in a bright doorway, strong backlighting",
        "two dancers in front of stage lights, silhouette",
        "a runner on the beach facing the low sun, silhouette",
        "a child standing by a bright curtain, backlit portrait",
        "a mountain hiker against the sun, silhouette",
    ],
    "white_bg": [
        "a white ceramic mug product photo on a pure white background",
        "a minimalist black logo centered on a pure white background",
        "a red lipstick product photo on a pure white background",
        "a pair of sunglasses on a seamless pure white background",
        "a toy robot isolated on a pure white background",
        "a simple line-art icon centered on a pure white background",
        "a perfume bottle product photo on a pure white background",
        "a wristwatch isolated on a seamless pure white background",
    ],
    "black_bg": [
        "a luxury watch product photo on a pure black background",
        "a glass perfume bottle on a pure black background, studio lighting",
        "a gold ring isolated on a pure black background",
        "a crystal glass on a pure black background with reflections",
        "a black camera lens product shot on a pure black background",
        "a silver necklace isolated on a pure black background",
        "a wine bottle studio shot on a pure black background",
        "a candle in darkness on a pure black background",
    ],
}

items = []
for category, prompts in groups.items():
    for i, prompt in enumerate(prompts, start=1):
        items.append({
            "id": f"{category}_{i:03d}",
            "category": category,
            "prompt": prompt,
            "target_brightness": (
                "dark" if category == "dark"
                else "bright" if category == "bright"
                else "neutral"
            ),
            "background_purity": (
                "white" if category == "white_bg"
                else "black" if category == "black_bg"
                else None
            ),
        })

OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"saved {len(items)} prompts to {OUT}")