from flask import Flask, request, render_template_string, jsonify
import torch
import requests
from transformers import BertTokenizer, GPT2Tokenizer, BertForSequenceClassification, GPT2LMHeadModel
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  # Allows us to run on Google Colab using Ngrok

# Initialize the tokenizer and models for both BERT and GPT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Google Custom Search API Details (Replace with your own API key and CX)
API_KEY = "AIzaSyB0OR4HjQtSA7v7nSVtChRNQ-7lMA0jfK4"
CSE_ID = "240d48d8b631c4b9e"

# Function to perform a custom search using Google Custom Search API
def google_search(query):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CSE_ID}"
    response = requests.get(search_url)
    search_results = response.json()
    return search_results['items'] if 'items' in search_results else []

# Function for Hybrid Summarization
def hybrid_summarize(input_text):
    # Extractive summarization using BERT
    inputs = bert_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        extractive_output = bert_model(**inputs)
    
    # Simplified: we select the entire input text (this could be improved by sentence-level selection)
    extracted_sentences = [input_text]
    
    # Abstractive summarization using GPT
    gpt_inputs = gpt_tokenizer(" ".join(extracted_sentences), return_tensors="pt")
    with torch.no_grad():
        summary_output = gpt_model.generate(**gpt_inputs, max_new_tokens=50)  # or max_length=100
    
    summary = gpt_tokenizer.decode(summary_output[0], skip_special_tokens=True)
    return summary

# HTML template as a string (Inline version)
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Custom Search Summarization Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        textarea, input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            border-radius: 5px;
            width: 100%;
        }
        button:hover {
            background-color: #45a049;
        }
        .summary {
            margin-top: 30px;
        }
        .summary h3 {
            color: #333;
        }
        .summary p {
            font-size: 16px;
            line-height: 1.5;
            color: #555;
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Custom Search Summarization Tool</h1>

    <form method="POST" action="/summarize">
        <input type="text" name="query" placeholder="Enter your search query..." required>
        <button type="submit">Generate Summaries</button>
    </form>

    {% if query %}
        <h2>Search Query: {{ query }}</h2>
        <div class="summary">
            <h3>Extractive Summary</h3>
            <p>{{ extractive_summary }}</p>
            <h3>Abstractive Summary</h3>
            <p>{{ abstractive_summary }}</p>
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_content)  # Render HTML page

@app.route('/summarize', methods=["POST"])
def summarize():
    query = request.form['query']  # User query for search engine
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Get search results from custom search engine
    search_results = google_search(query)
    
    if not search_results:
        return jsonify({"error": "No search results found"}), 404
    
    # Extract text from the search results (e.g., snippets)
    text = " ".join([result['snippet'] for result in search_results])
    
    # Generate summaries
    extractive_summary = hybrid_summarize(text)
    abstractive_summary = hybrid_summarize(text)
    
    return render_template_string(html_content, query=query, extractive_summary=extractive_summary, abstractive_summary=abstractive_summary)

# Start the Flask app
if __name__ == "__main__":
    app.run()
