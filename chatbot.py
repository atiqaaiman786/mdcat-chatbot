import json
import streamlit as st
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    import faiss_cpu as faiss

# Load and merge JSON data
with open("combined_mdcat_qa.json", encoding="utf-8", errors="ignore") as f1, open("MDCAT_FAQs.json", encoding="utf-8", errors="ignore") as f2:
    data = json.load(f1) + json.load(f2)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(questions)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# ✅ Load the correct question-answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/minilm-uncased-squad2")

# Retrieval function
def get_context(query, k=3, threshold=0.75):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    contexts = []
    for i, dist in zip(indices[0], distances[0]):
        if dist < threshold:
            contexts.append(answers[i])
    return contexts

# Generate answer using extractive QA pipeline
def generate_answer(query):
    context_list = get_context(query)
    if context_list:
        context = "\n".join(context_list)
        result = qa_pipeline(question=query, context=context)
        return result["answer"]
    else:
        return "Sorry, I couldn’t find a relevant answer. Try asking something from MDCAT past papers or policy topics."

# --- Streamlit Chat Interface (NEW UI)
st.set_page_config(page_title="Ask-MDCAT Chatbot", layout="centered")
st.title("📘 Ask-MDCAT Chatbot")
st.markdown("Ask anything related to MDCAT past papers or MDCAT policies.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input box
user_input = st.chat_input("Ask your MDCAT-related question...")

if user_input:
    with st.spinner("Thinking..."):
        bot_reply = generate_answer(user_input)
    st.session_state.chat_history.append((user_input, bot_reply))

# Show messages in order (newest at bottom)
for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").markdown(user_msg)
    st.chat_message("assistant").markdown(bot_msg)
