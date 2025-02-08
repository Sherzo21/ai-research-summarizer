import requests

# Define API URL
API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Define parameters for the request
params = {
    "query": "artificial intelligence",
    "fields": "title,abstract,year,authors,citationCount",
    "limit": 100
}

# Make the API request
response = requests.get(API_URL, params=params)

# Check if the request was successful
if response.status_code == 200:
    papers = response.json()

    # Loop through the results and print paper details
    for paper in papers.get('data', []):  # Ensure 'data' key exists
        title = paper.get("title", "No Title")
        citations = paper.get("citationCount", 0)
        print(f"Title: {title}, Citations: {citations}")

else:
    print("Failed to fetch data:", response.status_code, response.text)
