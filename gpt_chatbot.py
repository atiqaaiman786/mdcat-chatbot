import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Set your OpenAI API key (recommended via environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")  # or replace with "sk-..." string directly

# === Load Data ===
def load_data():
    data = []
    for file in ['combined_mdcat_qa.json', 'MDCAT_FAQs.json']:
        with open(file, 'r', encoding='utf-8') as f:
            data += json.load(f)
    return data

# === Vector Store Setup ===
def create_or_load_index(data, model, index_file='vector_store/mdcat.index'):
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        with open('vector_store/id_map.json', 'r') as f:
            id_map = json.load(f)
    else:
        embeddings = model.encode([d["question"] for d in data])
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        os.makedirs('vector_store', exist_ok=True)
        faiss.write_index(index, index_file)
        id_map = {str(i): d for i, d in enumerate(data)}
        with open('vector_store/id_map.json', 'w') as f:
            json.dump(id_map, f)
    return index, id_map

# === Semantic Search ===
#def search_query(query, model, index, id_map, top_k=1):
#    query_vector = model.encode([query])
#    D, I = index.search(np.array(query_vector).astype('float32'), top_k)
#    if I[0][0] == -1:
#        return None
#    return id_map[str(I[0][0])]["answer"]



def search_query(query, model, index, id_map, top_k=1, threshold=0.6):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector).astype('float32'), top_k)
    if I[0][0] == -1 or D[0][0] > threshold:  # lower distance = higher similarity
        return None
    return id_map[str(I[0][0])]["answer"]


# === OpenAI GPT Response ===
def generate_response_with_gpt(query):
    prompt = (
    "You are an intelligent tutor for MDCAT preparation. "
    "You should prioritize answering questions from the MDCAT syllabus. "
    "If the question is not directly related to MDCAT, still provide a helpful general response.\n\n"
    f"Question: {query}\nAnswer:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant who helps MDCAT students by answering both exam-related and general academic queries clearly and politely."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=150
    )
    return response.choices[0].message["content"].strip()
