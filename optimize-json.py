import sys

sys.path.append(".")

import json
from pathlib import Path
from tqdm import tqdm
import click
from config import settings as conf

root = Path.cwd()
dataset = conf.active.dataset
detector = conf.active.detector
confidence = conf.unidet.detect.confidence
json_in_dir = root / f"data/{dataset}/{detector}/detect/{confidence}/json"
json_out_dir = root / f"data/{dataset}/{detector}/detect/{confidence}/json-optim"

print("Input:", json_in_dir.relative_to(root))
print("Output:", json_out_dir.relative_to(root))

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

bar = tqdm(total=conf[dataset].n_videos)

for file in json_in_dir.glob("**/*.json"):
    json_out_path = json_out_dir / file.parent.name / file.name
    data = {}

    with open(file, "r") as f:
        json_data = json.load(f)

    for frame, bundle in json_data.items():
        data.update(
            {
                frame: [
                    ([round(i) for i in box], round(confidence, 3), clazz)
                    for box, confidence, clazz in bundle
                ]
            }
        )

    json_out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_out_path, "w") as f:
        json.dump(data, f)

    bar.update(1)

bar.close()
