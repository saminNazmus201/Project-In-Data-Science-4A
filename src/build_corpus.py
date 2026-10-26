
import json
import time
from pathlib import Path
import wikipediaapi

INPUT = Path("data/raw/ASQA.json")
OUTPUT = Path("data/processed/wiki_pages.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

wiki = wikipediaapi.Wikipedia(
    user_agent="ProportionalRAGResearch/1.0 (academic project)",
    language="en",
)

titles = set()

for example in data["dev"].values():
    for page in example.get("wikipages", []):
        title = page.get("title")
        if title:
            titles.add(title)

print("Unique Wikipedia titles:", len(titles))

saved = 0
missing = 0
seen_urls = set()
with OUTPUT.open("w", encoding="utf-8") as out:
    for i, title in enumerate(sorted(titles), start=1):
        try:
            page = wiki.page(title)

            if not page.exists() or not page.text.strip():
                 missing += 1
                 continue

            canonical_url = page.fullurl

            if canonical_url in seen_urls:
                print(f"Skipping duplicate page: {page.title}")
                continue

            seen_urls.add(canonical_url)

            record = {
                "title": page.title,
                 "url": canonical_url,
                "text": page.text,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1

            if i % 50 == 0:
                print(f"Processed {i}/{len(titles)}; saved={saved}")

            time.sleep(0.1)

        except Exception as exc:
            print(f"Failed: {title} — {exc}")

print("\nCorpus download complete")
print("Pages saved:", saved)
print("Pages missing/empty:", missing)
print("Output:", OUTPUT)
