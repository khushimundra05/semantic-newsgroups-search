# Semantic News Groups Search Project

A powerful search tool for discovering and filtering news articles across multiple group categories. Efficiently query, organize, and access news content with an intuitive interface.

# Tech Stack & Models:

Python, FastAPI, ChromaDB, Sentence Transformers, NumPy, Scikit-Fuzzy

# Steps and Explainantion

### 1.Virtual Environment

A python virtual environment was created to build and run the project. Reasons:

- Prevents version conflicts
- Ensures reproducibility
- Makes deployment easier as dependencies are clearly defined
- Keeps project libraries isolated from the global python installation

### 2.Dataset

The 20newsgroups dataset is used.

- Loaded using the fetch_20newsgroups() function
- This returns a Bunch object - similar to python dictionary but allows attribute access.
- Various attributes are extracted: documents, labels, categories.

documents = newsgroups.data  
labels = newsgroups.target  
categories = newsgroups.target_names

### 3. Embeddings - all-MiniLM-L6-v2 model

- We convert text to numerical vectors using embeddings
- all-MiniLM-L6-v2 model is a fast inference, small model widely used for semantic similarity tasks
- available under sentence-transformers library
- Each document becomes a 384 dimension vector
- Using `.encode()` the embeddings are generated
- Output shape: 11314 × 384

### 4. Vector Database - ChromaDB

- we use a vector database as it enables semantic (context based) search rather than just keyword search
- ChromaDB is lightweight, requires no external server, supports cosine similarity and has persistent storage
- A collection is created and documents are stored along with embeddings and metadata
- metadata contains information such as category, document id and cluster information
- Batch insert is used (batch size = 5000) as ChromaDB limits number of records per insert

### 5. Fuzzy C- Means Clustering

- Fuzzy C-means assigns each document a probability distribution across clusters instead of a single cluster
- we transpose the embeddings matrix as Fuzzy C-Means expects input as features × samples
- the algorithm returns cluster centers, membership matrix and FPC score
- FPC score indicates cluster quality where values closer to 1 indicate better separation
- we update the metadata after running fuzzy c means by storing the dominant cluster and membership vector

### 6. Semantic Cache

- We use caching to store results to avoid recomputation
- This decreases latency and CPU usage
- a semantic cache is used where embeddings are compared instead of matching text
- cosine similarity is used to compare query embeddings
- a threshold of similarity = 0.85 is used to determine if a cached query can be reused
- the cache is also cluster-aware so lookup is restricted to relevant clusters

### 7. FastAPI backend

- FastAPI is used to expose the search functionality as an API
- Request model is defined using

class QueryRequest(BaseModel):
query: str

- All API endpoints are exposed using FastAPI

# API endpoints

GET /

- basic endpoint to confirm the API is running

POST /query

- accepts a natural language query
- generates query embedding
- checks semantic cache
- if cache miss occurs the vector database is queried and the result is stored in cache

GET /cache/stats

- returns cache statistics including total entries, hit count, miss count and hit rate

DELETE /cache

- clears the cache and resets statistics

# How to run

Create virtual environment
python -m venv venv

Activate
venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Run data preparation
python data_prep.py

Run clustering
python clustering.py

Start API
uvicorn main:app --reload

Open
http://127.0.0.1:8000/docs
