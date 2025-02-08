from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Function to predict citation necessity
def recommend_citation(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    score = torch.softmax(outputs.logits, dim=1)
    
    # Prediction results
    citation_needed = torch.argmax(score, dim=1).item()
    confidence = score[0, citation_needed].item()
    
    return "Citation Recommended" if citation_needed == 1 else "No Citation Needed", confidence

# Example citation matching
paper_text = "This paper discusses deep learning in medical imaging."
recommendation, confidence = recommend_citation(paper_text)

print(f"Text: {paper_text}")
print(f"Recommendation: {recommendation} (Confidence: {confidence:.2f})")
