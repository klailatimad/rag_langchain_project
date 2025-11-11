import os, uuid, time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import chromadb
from chromadb.config import Settings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# ---------- Constants ----------
PERSIST_DIR = "./chroma_db"
RAG_COLLECTION = "mini"           # docs/knowledge chunks
CHAT_COLLECTION = "conversations" # chat transcripts + chat metadata
K = 3


# ---------- Chroma clients / collections ----------
@st.cache_resource
def get_chroma_client():
    return chromadb.Client(Settings(anonymized_telemetry=False,
                                    persist_directory=PERSIST_DIR))

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource
def get_vectorstore():
    chroma_client = get_chroma_client()       # ← get cached client
    embeddings = get_embeddings()             # ← get cached embeddings
    return Chroma(client=chroma_client,
                  collection_name=RAG_COLLECTION,
                  embedding_function=embeddings)

@st.cache_resource
def get_chat_collection():
    chroma_client = get_chroma_client()
    return chroma_client.get_or_create_collection(CHAT_COLLECTION)

@st.cache_resource
def build_rag_chain():
    vstore = get_vectorstore()              # <- pull from cache
    retriever = vstore.as_retriever(search_kwargs={"k": K})
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    contextualize = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's latest question into a standalone query using the chat history. Do not answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    hist_aware = create_history_aware_retriever(llm, retriever, contextualize)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use ONLY the provided context to answer. If not in the context, say you don't know. Keep answers concise (<=3 sentences).\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(hist_aware, qa_chain)

# ---------- Corpus helpers ----------
def ensure_demo_docs(vstore, embeddings):
    # load once if empty
    col = vstore._collection
    if col.count() == 0:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
        demo = [
            ("doc1", "LangChain is a framework for building LLM applications with composable primitives."),
            ("doc2", "RAG retrieves context chunks from a vector store and injects them into prompts.")
        ]
        docs, metas = [], []
        for sid, txt in demo:
            for d in splitter.create_documents([txt], metadatas=[{"source": sid}]):
                docs.append(d.page_content)
                metas.append(d.metadata)
        vstore.add_texts(docs, metadatas=metas)


# ---------- Chat persistence (Chroma) ----------
def list_chats(chat_col):
    try:
        # Chroma 1.x operator-style filter
        res = chat_col.get(
            where={"$and": [{"type": {"$eq": "chat_meta"}}]},
            include=["metadatas", "documents"]
        )
    except Exception:
        # Fallback for 0.4.x
        res = chat_col.get(
            where={"type": "chat_meta"},
            include=["metadatas", "documents"]
        )

    ids   = res.get("ids", [])
    metas = res.get("metadatas", [])
    items = []
    for i, meta in enumerate(metas):
        items.append({
            "id": ids[i],
            "name": meta.get("name", ids[i]),
            "created_at": meta.get("created_at"),
        })
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


def create_chat(chat_col, name=None):
    cid = str(uuid.uuid4())
    meta = {
        "type": "chat_meta",
        "name": name or f"Chat {cid[:8]}",
        "created_at": datetime.utcnow().isoformat()
    }
    # store minimal doc (embedding over the title for simplicity)
    chat_col.add(ids=[cid], documents=[meta["name"]], metadatas=[meta])
    return cid, meta["name"]

def save_message(chat_col, embeddings, chat_id, role, content, turn_index):
    # store each message as a doc (embedded), include turn for ordering
    mid = f"{chat_id}:{turn_index}:{role}"
    meta = {"type": "chat_message", "chat_id": chat_id, "role": role, "turn": turn_index, "ts": time.time()}
    # vec = embeddings.embed_query(content)  # short text -> cheap
    # chat_col.add(ids=[mid], documents=[content], embeddings=[vec], metadatas=[meta])
    chat_col.add(ids=[mid], documents=[content], metadatas=[meta])

def load_messages(chat_col, chat_id):
    try:
        # Chroma 1.x operator-style filter (single top-level operator)
        res = chat_col.get(
            where={"$and": [
                {"type": {"$eq": "chat_message"}},
                {"chat_id": {"$eq": chat_id}}
            ]},
            include=["metadatas", "documents"]
        )
    except Exception:
        # Fallback for 0.4.x
        res = chat_col.get(
            where={"type": "chat_message", "chat_id": chat_id},
            include=["metadatas", "documents"]
        )

    ids   = res.get("ids", [])
    docs  = res.get("documents", [])
    metas = res.get("metadatas", [])
    msgs = []
    for i, meta in enumerate(metas):
        msgs.append({
            "id": ids[i],
            "role": meta.get("role", "assistant"),
            "turn": meta.get("turn", 0),
            "content": docs[i]
        })
    msgs.sort(key=lambda m: m["turn"])
    return msgs



# ---------- UI helpers ----------
def ensure_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Add it to your environment or a .env file.")
        st.stop()

def render_chat(messages):
    for m in messages:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.write(m["content"])

def to_lc_history(messages):
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        else:
            out.append(AIMessage(content=m["content"]))
    return out


# ---------- App ----------
st.set_page_config(page_title="Local RAG — Chroma + OpenAI", page_icon="🔎", layout="wide")
st.title("🔎 Local RAG — Chroma + OpenAI")

ensure_api_key()
chroma_client = get_chroma_client()
embeddings = get_embeddings()
vstore = get_vectorstore()
chat_col = get_chat_collection()
rag = build_rag_chain()

# make sure we have *some* RAG content
ensure_demo_docs(vstore, embeddings)

def delete_chat_messages(chat_col, chat_id: str):
    # Chroma 1.x
    try:
        chat_col.delete(where={
            "$and": [
                {"type": {"$eq": "chat_message"}},
                {"chat_id": {"$eq": chat_id}}
            ]
        })
        return
    except Exception:
        # Chroma 0.4.x fallback
        chat_col.delete(where={"type": "chat_message", "chat_id": chat_id})

def delete_chat_fully(chat_col, chat_id: str):
    """Delete both messages AND the chat_meta row for this chat."""
    # 1) delete all messages for this chat (metadata filter)
    delete_chat_messages(chat_col, chat_id)

    # 2) delete the chat_meta row by ID (IDs are not filterable via 'where')
    try:
        chat_col.delete(ids=[chat_id])  # <-- this removes the meta doc created with id=chat_id
    except Exception:
        # Fallback: some older data might have stored name-only; try metadata match
        try:
            chat_col.delete(where={"$and": [{"type": {"$eq": "chat_meta"}}, {"id": {"$eq": chat_id}}]})
        except Exception:
            pass



# Sidebar: chat manager
with st.sidebar:
    st.header("Chats")
    chats = list_chats(chat_col)
    labels = [f'{c["name"]}' for c in chats]
    ids = [c["id"] for c in chats]

    # create chat if none exist
    if not ids:
        cid, cname = create_chat(chat_col, "Demo Chat")
        chats = list_chats(chat_col)
        labels = [c["name"] for c in chats]
        ids = [c["id"] for c in chats]

    # active chat in session
    default_index = 0
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = ids[0]
        st.session_state.turn = 0

    # selector
    idx = st.selectbox("Select a chat", options=list(range(len(ids))),
                       format_func=lambda i: labels[i], index=ids.index(st.session_state.chat_id) if st.session_state.chat_id in ids else 0)

    selected_chat_id = ids[idx]
    if selected_chat_id != st.session_state.chat_id:
        st.session_state.chat_id = selected_chat_id
        # reset turn (we’ll compute from persisted messages)
        st.session_state.turn = 0

    new_name = st.text_input("New chat name", value="", placeholder="Team Sync, Docs QA, ...")
    if st.button("➕ New Chat"):
        cid, cname = create_chat(chat_col, new_name or None)
        st.session_state.chat_id = cid
        st.session_state.turn = 0
        st.rerun()

    if st.button("🧹 Clear This Chat"):
        delete_chat_messages(chat_col, st.session_state.chat_id)
        st.session_state.turn = 0
        st.rerun()

    if st.button("🗑️ Delete Chat"):
        delete_chat_fully(chat_col, st.session_state.chat_id)

        # Repoint to a remaining chat (or create a new empty one), then rerun
        remaining = list_chats(chat_col)
        if remaining:
            st.session_state.chat_id = remaining[0]["id"]
        else:
            new_id, _ = create_chat(chat_col, "New Chat")
            st.session_state.chat_id = new_id

        st.session_state.turn = 0
        st.rerun()

# Load messages for active chat
messages = load_messages(chat_col, st.session_state.chat_id)
# compute next turn index
if messages:
    st.session_state.turn = messages[-1]["turn"] + 1
else:
    st.session_state.turn = 0
    # if empty, seed an assistant greeting (persist it)
    greet = "Hi! Ask me something about the indexed docs."
    save_message(chat_col, embeddings, st.session_state.chat_id, "assistant", greet, st.session_state.turn)
    st.session_state.turn += 1
    messages = load_messages(chat_col, st.session_state.chat_id)

# Render messages (shows both user + assistant)
render_chat(messages)

# Chat input
user_input = st.chat_input("Type your question")
if user_input:
    # save + render user message
    save_message(chat_col, embeddings, st.session_state.chat_id, "user", user_input, st.session_state.turn)
    st.session_state.turn += 1
    with st.chat_message("user"):
        st.write(user_input)

    # build LC history from persisted messages (including the one just saved)
    current_msgs = load_messages(chat_col, st.session_state.chat_id)
    lc_history = to_lc_history([m for m in current_msgs if m["role"] in ("user", "assistant")][:-1])  # exclude the last user msg for "history"

    # RAG invoke
    with st.chat_message("assistant"):
        try:
            resp = rag.invoke({"chat_history": lc_history, "input": user_input})
            answer = resp.get("answer", "")
            st.write(answer)

            # sources footer
            docs_used = resp.get("context", [])
            if docs_used:
                srcs = []
                for d in docs_used:
                    s = d.metadata.get("source", "unknown")
                    if s not in srcs:
                        srcs.append(s)
                st.caption("Sources: " + ", ".join(srcs))

            # persist assistant message
            save_message(chat_col, embeddings, st.session_state.chat_id, "assistant", answer, st.session_state.turn)
            st.session_state.turn += 1
        except Exception as e:
            st.error(f"Error: {e}")
