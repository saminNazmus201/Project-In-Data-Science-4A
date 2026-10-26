
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "asqa_wikipedia"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

model = SentenceTransformer(MODEL_NAME)

question = "Who has scored the most goals in international football?"

query_embedding = model.encode(
    question,
    normalize_embeddings=True,
).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)

print("Question:", question)
print("Collection size:", collection.count())
print("\nTop 5 retrieved passages:\n")

for i, (document, metadata, distance) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ),
    start=1,
):
    print(f"--- Result {i} ---")
    print("Title:", metadata["title"])
    print("Distance:", round(distance, 4))
    print("Text:", document[:500].replace("\n", " "))
    print()
