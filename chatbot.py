import json
import streamlit as st
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    import faiss_cpu as faiss

# Load JSON files
with open("combined_mdcat_qa.json", encoding="utf-8", errors="ignore") as f1, \
     open("MDCAT_FAQs.json", encoding="utf-8", errors="ignore") as f2:
    data = json.load(f1) + json.load(f2)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Embedding model for similarity search
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(questions)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# Query FAISS to get context
def get_context(query, k=3, threshold=0.75):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    contexts = []
    for i, dist in zip(indices[0], distances[0]):
        if dist < threshold:
            contexts.append(answers[i])
    return contexts

# --- ✅ Replace GPT-Neo with Ollama ---
def generate_with_ollama(prompt, model="llama3"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"⚠️ Ollama Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Ollama Exception: {str(e)}"

# Main answer generation
def generate_answer(query):
    context_list = get_context(query)
    if context_list:
        context = "\n".join(context_list)
        prompt = f"""You are a helpful MDCAT assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""
    else:
        prompt = f"""You are a knowledgeable MDCAT assistant. Answer the following question using your base knowledge.

Question: {query}
Answer:"""

    return generate_with_ollama(prompt)

# --- Streamlit UI
st.set_page_config(page_title="Ask-MDCAT Chatbot", layout="centered")
st.title("📘 Ask-MDCAT Chatbot")
st.markdown("Ask anything related to MDCAT past papers or policies.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask your MDCAT-related question...")

if user_input:
    with st.spinner("Thinking..."):
        bot_reply = generate_answer(user_input)
    st.session_state.chat_history.append((user_input, bot_reply))

for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").markdown(user_msg)
    st.chat_message("assistant").markdown(bot_msg)
