import chromadb
import numpy as np
import skfuzzy as fuzz

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path="./chromadb")

collection = client.get_collection("newsgroups_collection")

print("Fetching embeddings from ChromaDB...")

data = collection.get(include=["embeddings", "metadatas"])

embeddings = np.array(data["embeddings"])

print("Embeddings shape:", embeddings.shape)

# Transpose because Fuzzy C-Means expects (features × samples)
embedding_matrix = embeddings.T

n_clusters = 15

print("Running Fuzzy C-Means clustering...")

cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
    embedding_matrix,
    c=n_clusters,
    m=2,
    error=0.005,
    maxiter=1000,
    init=None
)

print("FPC score:", fpc)

memberships = u.T

print("Updating metadata with cluster memberships...")

updated_metadata = []

for i, meta in enumerate(data["metadatas"]):

    dominant_cluster = int(np.argmax(memberships[i]))

    meta["cluster"] = dominant_cluster
    meta["membership"] = memberships[i].tolist()

    updated_metadata.append(meta)

batch_size = 5000

for i in range(0, len(data["ids"]), batch_size):

    collection.update(
        ids=data["ids"][i:i+batch_size],
        metadatas=updated_metadata[i:i+batch_size]
    )

    print(f"Updated batch {i} → {min(i+batch_size, len(data['ids']))}")

print("Cluster memberships stored successfully.")