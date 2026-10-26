import json
from pathlib import Path

path = Path("data/processed/wiki_pages.jsonl")

total_pages = 0
total_chars = 0
lengths = []

with path.open("r", encoding="utf-8") as f:
    for line in f:
        page = json.loads(line)
        text = page.get("text", "")

        total_pages += 1
        total_chars += len(text)
        lengths.append(len(text))

lengths.sort()

print("Total pages:", total_pages)
print("Total characters:", total_chars)
print("Average characters per page:", round(total_chars / total_pages))
print("Median characters per page:", lengths[total_pages // 2])
print("Shortest page:", lengths[0])
print("Longest page:", lengths[-1])