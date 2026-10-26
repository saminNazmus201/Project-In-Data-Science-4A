
import json
from pathlib import Path

path = Path("data/raw/ASQA.json")

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

sample = data["dev"]["0"]

print("Top-level fields:")
print(list(sample.keys()))

print("\nQuestion:")
print(sample.get("ambiguous_question"))

for field in ["qa_pairs", "wikipages", "annotations"]:
    value = sample.get(field)
    print(f"\n{field}:")
    if isinstance(value, list):
        print("Count:", len(value))
        if value:
            print("First item keys:",
                  list(value[0].keys())
                  if isinstance(value[0], dict)
                  else type(value[0]))
            print("First item preview:", str(value[0])[:700])
    else:
        print(str(value)[:700])


print("\n--- Knowledge field inspection ---")

nonempty_knowledge = 0

for sample_id, example in data["dev"].items():
    for annotation in example.get("annotations", []):
        knowledge = annotation.get("knowledge", [])

        if knowledge:
            nonempty_knowledge += 1

            print("Sample ID:", sample_id)
            print("Knowledge item count:", len(knowledge))
            print("First knowledge item:", knowledge[0])
            print("Knowledge item type:", type(knowledge[0]).__name__)
            break

    if nonempty_knowledge:
        break

if nonempty_knowledge == 0:
    print("No non-empty knowledge found in dev annotations.")


total_examples = len(data["dev"])
examples_with_knowledge = 0
total_knowledge_items = 0

for sample_id, example in data["dev"].items():
    found_knowledge = False

    for annotation in example.get("annotations", []):
        knowledge = annotation.get("knowledge", [])

        if knowledge:
            total_knowledge_items += len(knowledge)
            found_knowledge = True

    if found_knowledge:
        examples_with_knowledge += 1

print("\n--- Knowledge statistics ---")
print("Total dev examples:", total_examples)
print("Examples with knowledge:", examples_with_knowledge)
print("Total knowledge items across annotations:", total_knowledge_items)