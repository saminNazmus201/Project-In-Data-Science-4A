
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
OUTPUT_PATH = Path("data/processed/embeddings.npy")
MANIFEST_PATH = Path("data/processed/embedding_manifest.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32

# Load the chunks in their existing order
texts = []

with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        texts.append(chunk["text"])

print("Total chunks:", len(texts))
print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

# Save embeddings as float32 to reduce storage
embeddings = embeddings.astype(np.float32)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_PATH, embeddings)

manifest = {
    "model_name": MODEL_NAME,
    "total_chunks": len(texts),
    "embedding_dimension": int(embeddings.shape[1]),
    "normalized": True,
    "batch_size": BATCH_SIZE,
    "chunks_file": str(CHUNKS_PATH),
    "embeddings_file": str(OUTPUT_PATH),
}

with MANIFEST_PATH.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nEmbedding complete!")
print("Embedding shape:", embeddings.shape)
print("Embedding file:", OUTPUT_PATH)
print("Manifest file:", MANIFEST_PATH)
print("Embedding file size (MB):",
      round(OUTPUT_PATH.stat().st_size / (1024 ** 2), 2))
