# 🧠 Local RAG Chat App (LangChain + Chroma + Streamlit)

An experimental Retrieval-Augmented Generation (RAG) chat application built with **LangChain**, **Chroma**, and **Streamlit** — running entirely **locally** with support for OpenAI models.

This proof of concept demonstrates how to:
- Build a minimal RAG pipeline with LangChain and OpenAI.
- Store and retrieve chat history using Chroma.
- Run a lightweight Streamlit front end for interactive conversations.

---

## 🚀 Current Capabilities

### ✅ RAG Core
- Context-aware retrieval using **Chroma** as a local vector store.
- Uses **OpenAI GPT-3.5-Turbo** for reasoning and answering.
- Simple, reusable chain:  
  - Contextualizes queries using chat history.  
  - Retrieves top-k document chunks.  
  - Generates concise context-aware answers.

### 💬 Streamlit Chat UI
- Minimal chat interface for asking and answering questions.
- Persists messages and metadata in Chroma.
- Supports creating, renaming, clearing, and deleting chats.
- Sidebar for switching between saved conversations.

### 💾 Local Storage
- All embeddings, metadata, and chat messages stored in ChromaDB locally.
- No cloud dependencies beyond the OpenAI API call.

---

## 🧩 Tech Stack

| Layer | Technology | Purpose |
|-------|-------------|----------|
| Front End | **Streamlit** | Interactive local web app |
| LLM | **OpenAI GPT-3.5-Turbo** | Chat & reasoning |
| Embeddings | **OpenAIEmbeddings** | Vector representation of docs |
| Vector Store | **ChromaDB** | Local semantic retrieval |
| Framework | **LangChain** | Chains, retrievers, and prompts |
| Environment | **Python 3.10+**, `venv` | Isolated local setup |

---

## ⚙️ Setup & Run

### 1️⃣ Clone and create environment
```bash
git clone <your-repo-url>
cd rag_langchain_project
python -m venv .venv
source .venv/bin/activate
```
2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Configure your environment
Create a .env file with your OpenAI key:
```OPENAI_API_KEY=sk-...```
4️⃣ Run the app
```bash
streamlit run app_streamlit.py
```
The app will open at http://localhost:8501

📁 Repository Structure
```
rag_langchain_project/
├── app_streamlit.py    # Streamlit frontend (chat UI + RAG logic)
├── requirements.txt
├── .env.example
├── .gitignore
└── chroma_db/          # Local Chroma vector storage (ignored in git)
```

🧭 Next Steps (Planned Roadmap)
|Priority|	Feature|	Description|
|---|---|---|
|🔥 High|	Persistent chat storage (SQLite/Postgres)|Chats and messages survive app restarts|
|🔥 High|	Document ingestion (PDFs, text, markdown)|	Upload, chunk, and embed real documents|
|🧩 Medium|	Retrieval controls|	Adjust k, chunk size, overlap, debug panel|
|🧩 Medium|	Streaming responses	|Real-time token streaming in UI|
|🧩 Medium	|Separation of front-end & back-end	|Streamlit → FastAPI + REST endpoints|
|🧪 Low|	Evaluation harness	|Measure latency, accuracy, token usage|
|🧱 Low|Auth & multi-user|	Separate users and their vector stores|
|⚙️ Low	|LangChain 1.x migration	|Move to new langchain-chroma and langchain-core APIs|

🧠 Key Concepts
- RAG (Retrieval-Augmented Generation): Enhances an LLM by retrieving relevant text chunks and injecting them into the prompt context.
- Chroma: A lightweight, open-source vector database for storing and searching embeddings locally.
- LangChain: A modular framework for chaining together LLMs, retrievers, and prompts into robust applications.