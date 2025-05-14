import json
import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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

# Load GPT-Neo 125M
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Context retriever
def get_context(query, k=3, threshold=0.75):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    contexts = []
    for i, dist in zip(indices[0], distances[0]):
        if dist < threshold:
            contexts.append(answers[i])
    return contexts

# Answer generator
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

    response = generator(prompt, max_length=300, do_sample=True)[0]['generated_text']
    return response.split("Answer:")[-1].strip()

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
