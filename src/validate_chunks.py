
import json
from pathlib import Path
from transformers import AutoTokenizer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
max_length = 256

total = 0
over_limit = 0
max_observed = 0

with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        encoded = tokenizer(
            chunk["text"],
            add_special_tokens=True,
            truncation=False,
        )
        token_count = len(encoded["input_ids"])

        total += 1
        max_observed = max(max_observed, token_count)

        if token_count > max_length:
            over_limit += 1

print("Total chunks checked:", total)
print("Model maximum sequence length:", max_length)
print("Longest observed input:", max_observed)
print("Chunks exceeding limit:", over_limit)
