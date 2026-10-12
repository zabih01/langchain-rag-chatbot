
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env
load_dotenv()

# Project paths
DOCUMENTS_DIR = Path("documents")
VECTORSTORE_DIR = Path("chroma_db")

# Gemini configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")


def create_vectorstore():
    """Load PDFs, split their text, and store embeddings in ChromaDB."""

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Create the '{DOCUMENTS_DIR}' folder and add a PDF."
        )

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{DOCUMENTS_DIR}'."
        )

    documents = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    # Remove empty pages
    documents = [
        doc for doc in documents
        if doc.page_content.strip()
    ]

    if not documents:
        raise ValueError(
            "No readable text found. Check whether your PDF is scanned."
        )

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    # Create local HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store vectors locally in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    print(f"Loaded {len(pdf_files)} PDF file(s).")
    print(f"Created {len(chunks)} text chunks.")

    return vectorstore


def load_vectorstore():
    """Load an existing ChromaDB vector store."""

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "Vector database not found. Run create_vectorstore() first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )


def ask_question(question: str, vectorstore=None) -> str:
    """Retrieve relevant document chunks and answer using Gemini."""

    if not question.strip():
        return "Please enter a question."

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to your .env file."
        )

    if vectorstore is None:
        vectorstore = load_vectorstore()

    # Step 1: Retrieval
    retrieved_docs = vectorstore.similarity_search(question, k=4)

    if not retrieved_docs:
        return "I couldn't find relevant information in your documents."

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown document')}, "
        f"page: {doc.metadata.get('page', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in retrieved_docs
    )

    # Step 2: Generation
    llm = init_chat_model(
        model=MODEL_NAME,
        model_provider="google_genai",
    )

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the question using the document context below.
If the answer is not contained in the context, clearly say that
you could not find the answer in the uploaded documents.
Do not invent facts.

Document context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    # Run this file once to index your PDFs.
    create_vectorstore()

    # Ask questions in the terminal.
    vectorstore = load_vectorstore()

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            break

        try:
            answer = ask_question(query, vectorstore)
            print(f"\nAnswer:\n{answer}")
        except Exception as exc:
            print(f"\nError: {exc}")