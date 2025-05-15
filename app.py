

import json
import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# === Chat UI ===
def main():
    #st.set_page_config(page_title="ASK MDCAT Assistant", page_icon="💬")
    st.markdown("<h1 style='text-align: center;'>💬 ASK MDCAT Assistant</h1>", unsafe_allow_html=True)
    st.write("Ask anything about past papers or MDCAT test policy.")

    # Inject chat bubble CSS
    st.markdown("""
        <style>
        .chat-bubble {
            padding: 10px 16px;
            margin: 6px 0;
            border-radius: 12px;
            max-width: 75%;
            font-size: 16px;
            display: inline-block;
        }
        .user {
            background-color: #0078D4;
            color: white;
            text-align: right;
            margin-left: auto;
        }
        .bot {
            background-color: #f0f2f6;
            color: black;
            text-align: left;
            border: 1px solid #ccc;
        }
        </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("", key="input_text", placeholder="Ask your question here...")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.chat_history.append(("You", user_input))

        # === Answer Logic ===
        best_answer = search_query(user_input, embed_model, faiss_index, index_map)
        if best_answer:
            response = best_answer
        else:
            prompt = f"MDCAT student asked: {user_input}\nAnswer:"
            response = llm(prompt, max_length=100, do_sample=True)[0]["generated_text"].split("Answer:")[-1].strip()

        st.session_state.chat_history.append(("Bot", response))

    # === Chat Rendering ===
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"<div class='chat-bubble user'>👤 {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble bot'>🤖 {msg}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    data = load_data()
    faiss_index, index_map = create_or_load_index(data, embed_model)
    llm = load_llm()
    main()