from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#Load and merge JSON data
#with open(r"C:\Users\DELL\Downloads\mdcat\mdcat_chatbot\combined_output.json", encoding="utf-8", errors="ignore") as f1, #open(r"C:\Users\DELL\Downloads\mdcat\mdcat_chatbot\MDCAT_FAQs.json", encoding="utf-8", errors="ignore") as f2:
#    data = json.load(f1) + json.load(f2)


with open(r"C:\Users\DELL\Downloads\mdcat\mdcat_chatbot\combined_mdcat_qa.json", encoding="utf-8", errors="ignore") as f1, \
     open(r"C:\Users\DELL\Downloads\mdcat\mdcat_chatbot\MDCAT_FAQs.json", encoding="utf-8", errors="ignore") as f2:
    data = json.load(f1) + json.load(f2)


questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(questions)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# Load Phi-2 LLM
#tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
#model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="auto")
#generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


# Retrieval function
def get_context(query, k=3, threshold=0.75):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    contexts = []
    for i, dist in zip(indices[0], distances[0]):
        if dist < threshold:
            contexts.append(answers[i])
    return contexts

# RAG prompt + fallback logic
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
        prompt = f"""You are a knowledgeable MDCAT assistant. Answer the following question using general knowledge if needed.

Question: {query}
Answer:"""

    output = generator(prompt, max_new_tokens=200)[0]["generated_text"]
    return output.split("Answer:")[-1].strip()

# Streamlit GUI
st.title("📘 MDCAT Chatbot")
user_query = st.text_input("Ask your MDCAT-related question:")

if user_query:
    with st.spinner("Thinking..."):
        response = generate_answer(user_query)
    st.markdown(f"**Answer:** {response}")
