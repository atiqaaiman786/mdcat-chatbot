import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# === Load Data ===
def load_data():
    data = []
    for file in ['data/past_papers.json', 'data/test_policies.json']:
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
def search_query(query, model, index, id_map, top_k=1):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector).astype('float32'), top_k)
    if I[0][0] == -1:
        return None
    return id_map[str(I[0][0])]["answer"]

# === Load LLM ===
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator
