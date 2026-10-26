
import json
from pathlib import Path
from transformers import AutoTokenizer

INPUT = Path("data/processed/wiki_pages.jsonl")
OUTPUT = Path("data/processed/chunks.jsonl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 240
CHUNK_OVERLAP = 50

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

total_articles = 0
total_chunks = 0

with INPUT.open("r", encoding="utf-8") as source, \
     OUTPUT.open("w", encoding="utf-8") as destination:

    for line in source:
        article = json.loads(line)
        text = article.get("text", "").strip()

        if not text:
            continue

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )

        offsets = encoded["offset_mapping"]
        token_count = len(offsets)

        if token_count == 0:
            continue

        total_articles += 1
        step = CHUNK_SIZE - CHUNK_OVERLAP

        for start in range(0, token_count, step):
            end = min(start + CHUNK_SIZE, token_count)

            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            chunk_text = text[char_start:char_end].strip()

            if not chunk_text:
                continue

            record = {
                "chunk_id": total_chunks,
                "title": article["title"],
                "url": article["url"],
                "text": chunk_text,
                "token_count": end - start,
            }

            destination.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            total_chunks += 1

            if end == token_count:
                break

print("Chunking complete!")
print("Articles processed:", total_articles)
print("Total chunks:", total_chunks)
print("Chunk size:", CHUNK_SIZE, "tokens")
print("Chunk overlap:", CHUNK_OVERLAP, "tokens")
print("Output:", OUTPUT)
