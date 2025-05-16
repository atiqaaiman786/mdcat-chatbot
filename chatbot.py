import json
import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# === Load data ===
def load_data():
    data = []
    for file in ['combined_mdcat_qa.json', 'MDCAT_FAQs.json']:
        with open(file, 'r', encoding='utf-8') as f:
            data += json.load(f)
    return data

# === Prepare vector DB ===
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

# === Get best match ===
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

# === Load LLM ===
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator


# === Custom Prompted Response Function ===
def generate_response_with_prompt(generator, query):
    prompt = (
        "You are an intelligent tutor for MDCAT preparation. "
        "You should prioritize answering questions from the MDCAT syllabus. "
        "If the question is not directly related to MDCAT, still provide a helpful general response.\n\n"
        f"Question: {query}\nAnswer:"
    )
    response = generator(prompt, max_length=200, temperature=0.7, do_sample=True)
    return response[0]["generated_text"].replace(prompt, "").strip()

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="ASK MDCAT Assistant", page_icon="💬")
    st.markdown("<h1 style='text-align: center;'>💬 ASK MDCAT Assistant</h1>", unsafe_allow_html=True)
    st.write("Ask anything about past papers or MDCAT test policy.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", key="input_text", placeholder="Ask your question here...")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.chat_history.append(("You", user_input))

        # Retrieval + LLM
        best_answer = search_query(user_input, embed_model, faiss_index, index_map)
        if best_answer:
            response = best_answer
        else:
            prompt = f"MDCAT student asked: {user_input}\nAnswer:"
            response = llm(prompt, max_length=100, do_sample=True)[0]["generated_text"].split("Answer:")[-1].strip()

        st.session_state.chat_history.append(("Bot", response))

    for sender, msg in st.session_state.chat_history[::-1]:
        st.markdown(f"**{sender}:** {msg}")

if __name__ == "__main__":
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    data = load_data()
    faiss_index, index_map = create_or_load_index(data, embed_model)
    llm = load_llm()
    main()