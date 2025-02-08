from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load pre-trained BERT model for sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample dataset: Titles of research papers
papers = [
    "Deep Learning for AI",
    "Transformer models for NLP",
    "Quantum AI",
    "Graph Neural Networks in Finance",
    "AI-based Drug Discovery"
]

# Convert papers to embeddings
embeddings = model.encode(papers)

# Initialize FAISS index (L2 distance search)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Function to search for similar papers
def search_paper(query, top_k=1):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    results = [papers[idx] for idx in indices[0]]
    return results

# Example search query
query = "AI for scientific papers"
best_match = search_paper(query)

# Print results
print(f"Query: {query}")
print(f"Best Match: {best_match}")
