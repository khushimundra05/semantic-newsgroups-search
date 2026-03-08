from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

from cache import SemanticCache

app = FastAPI()

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chromadb")

collection = client.get_collection("newsgroups_collection")

cache = SemanticCache()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "Semantic search API running"}


@app.post("/query")
def query_search(request: QueryRequest):

    query = request.query
    embedding = model.encode([query])[0]

    results = collection.query(query_embeddings=[embedding.tolist()], n_results=1)

    dominant_cluster = results["metadatas"][0][0]["cluster"]

    cached = cache.lookup(embedding, dominant_cluster)

    if cached:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached["matched_query"],
            "similarity_score": cached["similarity"],
            "result": cached["result"],
            "dominant_cluster": cached["cluster"],
        }

    results = collection.query(query_embeddings=[embedding.tolist()], n_results=1)

    document = results["documents"][0][0]

    cache.store(query, embedding, document, dominant_cluster)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": document,
        "dominant_cluster": dominant_cluster,
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}
