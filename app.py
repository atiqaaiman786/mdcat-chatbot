# Streamlit UI
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