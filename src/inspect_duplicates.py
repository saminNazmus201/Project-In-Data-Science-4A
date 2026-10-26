import json
from collections import defaultdict
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.jsonl")

text_to_chunks = defaultdict(list)

with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)

        normalized_text = " ".join(chunk["text"].split())

        text_to_chunks[normalized_text].append({
            "chunk_id": chunk["chunk_id"],
            "title": chunk["title"],
            "url": chunk.get("url", "N/A"),
        })

duplicate_groups = [
    chunks
    for chunks in text_to_chunks.values()
    if len(chunks) > 1
]

print("Total duplicate groups:", len(duplicate_groups))

print("\n--- First 10 duplicate examples ---")

for i, group in enumerate(duplicate_groups[:10], start=1):
    print(f"\nDuplicate group {i}:")
    print("Number of occurrences:", len(group))

    for chunk in group:
        print(
            f"  ID: {chunk['chunk_id']} | "
            f"Title: {chunk['title']}"
        )