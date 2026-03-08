from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

print("Loading dataset...")

newsgroups = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

documents = newsgroups.data
labels = newsgroups.target
categories = newsgroups.target_names

print(f"Loaded {len(documents)} documents")

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(documents, show_progress_bar=True)

# Create persistent ChromaDB client
print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path="./chromadb")

collection = client.get_or_create_collection(
    name="newsgroups_collection"
)

print("Preparing metadata...")

ids = [str(uuid.uuid4()) for _ in documents]

metadatas = [
    {
        "category": categories[labels[i]],
        "doc_id": i
    }
    for i in range(len(documents))
]

print("Storing documents in ChromaDB...")

batch_size = 5000

for i in range(0, len(documents), batch_size):

    collection.add(
        ids=ids[i:i+batch_size],
        embeddings=[e.tolist() for e in embeddings[i:i+batch_size]],
        documents=documents[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size]
    )

    print(f"Inserted batch {i} → {min(i+batch_size, len(documents))}")

print("Data stored successfully.")