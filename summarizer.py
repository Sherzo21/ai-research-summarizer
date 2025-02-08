from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="t5-small")

# Function to summarize text
def summarize_paper(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Example usage
if __name__ == "__main__":
    paper_abstract = """
    Artificial intelligence is rapidly advancing and influencing various fields such as healthcare, 
    finance, and autonomous systems. The ability of AI to process large amounts of data and learn from it 
    has led to significant improvements in automation, predictive analytics, and personalized recommendations.
    """
    summary = summarize_paper(paper_abstract)
    print("Summary:", summary)
