
import streamlit as st
from dotenv import load_dotenv
from rag import ask_question, load_vectorstore, create_vectorstore

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Chat with Your PDF")
st.caption("Ask questions about your documents using AI.")

with st.sidebar:
    st.header("Settings")

    st.markdown(
        """
        - **LLM:** Google Gemini
        - **Embeddings:** HuggingFace
        - **Vector Database:** ChromaDB
        """
    )

    if st.button("Rebuild Knowledge Base", use_container_width=True):
        try:
            with st.spinner("Processing PDFs..."):
                create_vectorstore()
                st.session_state.vectorstore = load_vectorstore()

            st.success("Knowledge base rebuilt successfully!")

        except Exception as exc:
            st.error(f"Error rebuilding knowledge base: {exc}")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    try:
        with st.spinner("Loading knowledge base..."):
            st.session_state.vectorstore = load_vectorstore()

    except FileNotFoundError:
        st.warning(
            "Knowledge base not found. Add PDF files to the "
            "documents folder and click Rebuild Knowledge Base."
        )
        st.stop()

    except Exception as exc:
        st.error(f"Could not load knowledge base: {exc}")
        st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your PDF...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                answer = ask_question(
                    question,
                    st.session_state.vectorstore,
                )
                st.markdown(answer)

            except Exception as exc:
                answer = f"Error: {exc}"
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )