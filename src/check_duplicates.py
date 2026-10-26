import json
from collections import Counter
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.jsonl")

chunk_ids = []
text_counts = Counter()
title_counts = Counter()

with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)

        chunk_ids.append(chunk["chunk_id"])

        # Compare complete chunk text, not just a preview
        text = " ".join(chunk["text"].split())
        text_counts[text] += 1

        # Count chunks belonging to each article
        title_counts[chunk["title"]] += 1

total_chunks = len(chunk_ids)

unique_texts = len(text_counts)
duplicate_text_groups = sum(
    1 for count in text_counts.values() if count > 1
)
extra_duplicate_chunks = sum(
    count - 1 for count in text_counts.values() if count > 1
)

print("Total chunks:", total_chunks)
print("Unique chunk texts:", unique_texts)
print("Duplicate text groups:", duplicate_text_groups)
print("Extra duplicate chunks:", extra_duplicate_chunks)
print("Unique chunk IDs:", len(set(chunk_ids)))
print("Unique article titles:", len(title_counts))

print("\nTop 5 articles by number of chunks:")

for title, count in title_counts.most_common(5):
    print(f"{count:4} chunks | {title}")