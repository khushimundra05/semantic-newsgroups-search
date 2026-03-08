import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.85):

        self.threshold = threshold
        self.cache = {}

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, embedding, cluster):

        if cluster not in self.cache:
            self.miss_count += 1
            return None

        entries = self.cache[cluster]

        for entry in entries:

            sim = cosine_similarity([embedding], [entry["embedding"]])[0][0]

            if sim >= self.threshold:

                self.hit_count += 1

                return {
                    "matched_query": entry["query"],
                    "similarity": float(sim),
                    "result": entry["result"],
                    "cluster": cluster,
                }

        self.miss_count += 1
        return None

    def store(self, query, embedding, result, cluster):

        entry = {"query": query, "embedding": embedding, "result": result}

        if cluster not in self.cache:
            self.cache[cluster] = []

        self.cache[cluster].append(entry)

    def stats(self):

        total = self.hit_count + self.miss_count

        return {
            "total_entries": sum(len(v) for v in self.cache.values()),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total else 0,
        }

    def clear(self):

        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
