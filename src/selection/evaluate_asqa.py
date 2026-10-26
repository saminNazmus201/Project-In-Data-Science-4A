import json
import random
from pathlib import Path

DATASET_PATH = Path("data/raw/ASQA.json")
OUTPUT_DIR = Path("data/results")
NUM_QUESTIONS = 50
RANDOM_SEED = 42


def load_asqa_questions():
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data["dev"]

    # Only use examples with a non-empty ambiguous_question.
    eligible = []

    for sample_id, example in examples.items():
        question = example.get("ambiguous_question")

        if isinstance(question, str) and question.strip():
            eligible.append({
                "sample_id": sample_id,
                "question": question.strip(),
            })

    if len(eligible) < NUM_QUESTIONS:
        raise ValueError(
            f"Need {NUM_QUESTIONS} questions, "
            f"but found only {len(eligible)}."
        )

    # Reproducible sampling: the same 50 questions each run.
    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(eligible, NUM_QUESTIONS)

    return selected


def main():
    questions = load_asqa_questions()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "asqa_50_questions.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Selected questions: {len(questions)}")
    print(f"Saved to: {output_path}")

    print("\nFirst 5 questions:")
    for item in questions[:5]:
        print(f'ID {item["sample_id"]}: {item["question"]}')


if __name__ == "__main__":
    main()