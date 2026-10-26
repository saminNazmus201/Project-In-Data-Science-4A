
import json
from pathlib import Path

import chromadb
import numpy as np

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
DB_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "asqa_wikipedia"

BATCH_SIZE = 500

print("Loading chunks...")

with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

print("Loading saved embeddings...")

embeddings = np.load(EMBEDDINGS_PATH)

if len(chunks) != len(embeddings):
    raise ValueError(
        f"Mismatch: {len(chunks)} chunks but "
        f"{len(embeddings)} embeddings"
    )

if embeddings.shape[1] != 384:
    raise ValueError(
        f"Expected 384-dimensional embeddings, "
        f"got {embeddings.shape[1]}"
    )

print("Connecting to persistent ChromaDB...")

client = chromadb.PersistentClient(path=DB_PATH)

# Create a fresh collection for this build.
# This avoids accidentally mixing old and new records.
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

print("Adding records to ChromaDB...")

for start in range(0, len(chunks), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(chunks))

    batch_chunks = chunks[start:end]
    batch_embeddings = embeddings[start:end]

    ids = [str(c["chunk_id"]) for c in batch_chunks]
    documents = [c["text"] for c in batch_chunks]
    metadatas = [
        {
            "title": c["title"],
            "url": c["url"],
            "chunk_id": int(c["chunk_id"]),
        }
        for c in batch_chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=batch_embeddings.tolist(),
    )

    print(f"Added {end}/{len(chunks)} chunks")

print("\nIndex build complete!")
print("Collection:", COLLECTION_NAME)
print("Stored records:", collection.count())
print("Database path:", DB_PATH)
