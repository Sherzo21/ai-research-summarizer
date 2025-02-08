from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, BertTokenizer, BertForSequenceClassification

# Initialize FastAPI
app = FastAPI()

# Load summarization model
summarizer = pipeline("summarization", model="t5-small")

# Load search model
search_model = SentenceTransformer("all-MiniLM-L6-v2")
papers = [
    "Deep Learning for AI",
    "Transformer models for NLP",
    "Quantum AI",
    "Graph Neural Networks in Finance",
    "AI-based Drug Discovery"
]
embeddings = search_model.encode(papers)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Load citation recommendation model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
citation_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Request model for input validation
class PaperRequest(BaseModel):
    text: str

# Summarization endpoint
@app.post("/summarize/")
def summarize(request: PaperRequest):
    summary = summarizer(request.text, max_length=150, min_length=50, do_sample=False)
    return {"summary": summary[0]['summary_text']}

# Search endpoint
@app.post("/search/")
def search_paper(request: PaperRequest):
    query_embedding = search_model.encode([request.text])
    _, indices = index.search(np.array(query_embedding), 1)
    return {"best_match": papers[indices[0][0]]}

# Citation recommendation endpoint
@app.post("/recommend/")
def recommend_citation(request: PaperRequest):
    inputs = tokenizer(request.text, return_tensors="pt")
    outputs = citation_model(**inputs)
    score = torch.softmax(outputs.logits, dim=1)
    
    citation_needed = torch.argmax(score, dim=1).item()
    confidence = score[0, citation_needed].item()
    
    return {
        "recommendation": "Citation Recommended" if citation_needed == 1 else "No Citation Needed",
        "confidence": round(confidence, 2)
    }

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
