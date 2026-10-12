# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Streamlit, Google Gemini, HuggingFace embeddings, and ChromaDB. It allows users to ask questions about PDF documents and receive AI-generated answers based on the document content.

## Features

- Chat with PDF documents.
- Extract and split PDF text into chunks.
- Generate embeddings using HuggingFace.
- Store and retrieve document embeddings using ChromaDB.
- Generate answers using Google Gemini.
- Interactive chat interface built with Streamlit.
- Maintain chat history during a session.
- Rebuild the knowledge base when documents change.

## Technologies Used

- Python
- LangChain
- Streamlit
- Google Gemini
- HuggingFace Sentence Transformers
- ChromaDB
- PyPDF
- uv

## Project Structure

```text
langchain-rag-chatbot/
├── app.py
├── rag.py
├── documents/
│   └── sample.pdf
├── chroma_db/
├── .env
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python 3.10 or newer
- uv package manager
- Google Gemini API key
- One or more text-based PDF documents

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/langchain-rag-chatbot.git
cd langchain-rag-chatbot
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
uv sync
```

If Streamlit has not been installed yet, run:

```bash
uv add streamlit
```

### 3. Configure Environment Variables

Create a `.env` file in the project root directory.

Add the following configuration:

```dotenv
GOOGLE_API_KEY=your_google_gemini_api_key
MODEL_NAME=gemini-2.5-flash
```

Replace `your_google_gemini_api_key` with your actual Google Gemini API key.

Keep your API key private. Do not commit your `.env` file to GitHub.

### 4. Add PDF Documents

Create a folder named `documents` in the project root.

Place your PDF files inside it:

```text
documents/
├── sample.pdf
└── another_document.pdf
```

Use valid, readable PDF files.

### 5. Build the Knowledge Base

Run:

```bash
uv run python rag.py
```

This processes the PDFs, splits their text into chunks, generates embeddings, and stores them in the local ChromaDB vector database.

When the terminal asks for a question, type `exit` to leave the terminal chatbot.

### 6. Start the Streamlit Application

Run:

```bash
uv run streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

You can now ask questions about your PDF documents through the web interface.

## How It Works

1. **Document Loading:** PDF files are loaded using PyPDFLoader.
2. **Text Splitting:** Documents are split into smaller chunks using RecursiveCharacterTextSplitter.
3. **Embedding Generation:** HuggingFace converts text chunks into vector embeddings.
4. **Vector Storage:** ChromaDB stores the embeddings locally.
5. **Retrieval:** The chatbot retrieves relevant chunks for a user's question.
6. **Answer Generation:** Google Gemini uses the retrieved context to generate an answer.
7. **User Interface:** Streamlit displays the conversation in a web interface.

## Configuration

The current RAG pipeline uses:

- Chunk size: 1000 characters
- Chunk overlap: 150 characters
- Retrieved chunks per question: 4
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Default LLM: `gemini-2.5-flash`

These settings can be changed in `rag.py`.

## Troubleshooting

### Google API Key Error

Make sure your `.env` file contains a valid `GOOGLE_API_KEY`.

### No PDF Files Found

Ensure your PDF files are inside the `documents` folder.

### Knowledge Base Not Found

Build the vector database by running:

```bash
uv run python rag.py
```

You can also use the **Rebuild Knowledge Base** button in the Streamlit sidebar.

### PDF Parsing Error

Make sure the file is a genuine, readable PDF rather than a text or Markdown file renamed with a `.pdf` extension.

### Port Already in Use

Run Streamlit on another port:

```bash
uv run streamlit run app.py --server.port 8502
```

## Security

- Never upload your `.env` file to GitHub.
- Never publish your API key.
- Do not commit private or confidential documents.
- Review document contents before sharing this project publicly.

## Future Improvements

- Upload PDF files directly through the Streamlit interface.
- Display source documents and page numbers with answers.
- Support multiple document collections.
- Improve document indexing and vector database updates.
- Add user authentication.

## License

This project is available for learning and educational purposes. Add a license file if you intend to distribute it publicly.
