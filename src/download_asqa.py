
from datasets import load_dataset
import json
from pathlib import Path

output_path = Path("data/raw/ASQA.json")

dataset = load_dataset("din0s/asqa")

data = {
    split: {
        str(i): example
        for i, example in enumerate(dataset[split])
    }
    for split in dataset.keys()
}

with output_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Dataset saved:", output_path)
print("Splits:", list(data.keys()))

for split, examples in data.items():
    print(f"{split}: {len(examples)} examples")
