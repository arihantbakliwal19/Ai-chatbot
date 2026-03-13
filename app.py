import streamlit as st
from models.llm import get_chatgroq_model
from utils.rag import load_documents, create_vector_db, retrieve_context
from utils.web_search import search_web

st.title("AI Career Assistant Chatbot")
uploaded_file = st.file_uploader("Upload a PDF to analyze", type=["pdf"])

if uploaded_file is not None:

    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    docs = load_documents("uploaded.pdf")
    st.session_state.vector_db = create_vector_db(docs)

    st.success("PDF uploaded successfully! The chatbot will now use this document.")

chat_model = get_chatgroq_model()

if "vector_db" not in st.session_state:

    docs = load_documents("docs/knowledge.pdf")
    st.session_state.vector_db = create_vector_db(docs)

vector_db = st.session_state.vector_db
mode = st.sidebar.selectbox(
    "Response Mode",
    ["Concise", "Detailed"]
)

if mode == "Concise":
    system_prompt = "Give short answers."
else:
    system_prompt = "Give detailed explanations."

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    context = retrieve_context(prompt, vector_db)
    web_info = search_web(prompt)

    final_prompt = f"""
    {system_prompt}

    DOCUMENT CONTEXT:
    {context}

    WEB INFO:
    {web_info}

    USER QUESTION:
    {prompt}
    """

    response = chat_model.invoke(final_prompt)

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.content}
    )