import streamlit as st
import os
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
from google.genai import Client
from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# Load Environment & Configure Gemini
# =========================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY missing.")
    st.stop()

client = Client(api_key=API_KEY)

EMBED_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-2.5-flash"


def embed_texts(texts):
    """Embed a list of strings into vectors using Gemini embeddings."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts
    )
    return [list(e.values) for e in response.embeddings]


# =========================================
# Streamlit Page Setup
# =========================================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Process PDF", "Chat Interface"], index=0)

# =========================================
# Session State
# =========================================
for key in ["chunks", "embeddings", "messages", "doc_stats", "doc_name"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else None


# =========================================
# HOME
# =========================================
if page == "Home":
    st.title("🤖 RAG Chatbot – PDF Knowledge AI")
    st.write("Upload a PDF → Process it → Chat using Gemini + RAG")


# =========================================
# PROCESS PDF
# =========================================
if page == "Process PDF":

    st.header("📄 Upload & Process Your Document")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    chunk_size = st.sidebar.slider("Chunk Size (words)", 300, 2000, 800, 100)
    overlap = st.sidebar.slider("Chunk Overlap (words)", 0, 400, 150, 50)

    if pdf and st.button("🚀 Process Now"):

        reader = PdfReader(pdf)
        pages = [p.extract_text() or "" for p in reader.pages]
        full_text = " ".join(pages)

        words = full_text.split()
        chunks = []
        i = 0

        while i < len(words):
            chunks.append(" ".join(words[i:i + chunk_size]))
            i += chunk_size - overlap

        if not chunks:
            st.error("❌ No text extracted. This PDF may be scanned.")
        else:
            embeddings = embed_texts(chunks)

            st.session_state.chunks = chunks
            st.session_state.embeddings = np.array(embeddings)
            st.session_state.doc_name = pdf.name
            st.session_state.messages = []

            st.session_state.doc_stats = {
                "Pages": len(pages),
                "Chunks": len(chunks),
                "Characters": len(full_text)
            }

            st.success(f"✅ Indexed {len(chunks)} chunks successfully!")

    if st.session_state.doc_stats:
        st.subheader("📊 Document Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pages", st.session_state.doc_stats["Pages"])
        c2.metric("Chunks", st.session_state.doc_stats["Chunks"])
        c3.metric("Characters", f"{st.session_state.doc_stats['Characters']:,}")


# =========================================
# CHAT INTERFACE
# =========================================
if page == "Chat Interface":

    st.header("💬 Chat with Your Document")

    if not st.session_state.chunks:
        st.warning("⚠ Please upload and process a PDF first.")
        st.stop()

    # ===== Chat History Rendering (KEEP THIS EXACTLY) =====
    for msg in st.session_state.messages:
        render_role = "assistant" if msg["role"] == "user" else "user"
        with st.chat_message(render_role):
            st.write(msg["content"])

    query = st.chat_input("Ask anything from the document...")

    if query:
        # USER MESSAGE (LEFT)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            st.write(query)

        # ===== RETRIEVAL =====
        q_emb = embed_texts([query])[0]
        sims = cosine_similarity([q_emb], st.session_state.embeddings)[0]
        idx = sims.argsort()[-3:][::-1]

        selected_chunks = [st.session_state.chunks[i] for i in idx]

        context = "\n\n".join(
            [f"--- CHUNK {i+1} ---\n{ch}" for i, ch in enumerate(selected_chunks)]
        )

        # ===== STRICT SYSTEM INSTRUCTION =====
        system_instruction = (
            "You are a strict document-based question answering assistant.\n"
            "You MUST answer ONLY the user's exact question.\n"
            "You MUST use ONLY the provided chunks.\n"
            "Do NOT add explanations, lists, interview questions, or extra details.\n"
            "If the answer is NOT found in the chunks, reply EXACTLY:\n"
             "I’m unable to answer this question because it falls outside the scope of the uploaded document."
        )

        user_prompt = f"""
CONTEXT:
{context}

QUESTION:
{query}

RULES:
- Answer only the question
- Be concise
- No external knowledge
"""

        # ===== GEMINI (FIXED FORMAT) =====
        full_prompt = f"""
SYSTEM:
{system_instruction}

======================

{user_prompt}
"""

        response = client.models.generate_content(
            model=GEN_MODEL,
            contents=full_prompt
        )

        answer = response.text

        # ASSISTANT MESSAGE (RIGHT)
        with st.chat_message("user"):
            st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # ===== TERMINAL LOGGING (MENTOR REQUIRED) =====
        print("\n" + "=" * 80)
        print("USER QUESTION:")
        print(query)

        print("\nRETRIEVED CHUNKS:")
        for i, ch in enumerate(selected_chunks, start=1):
            print(f"\n--- CHUNK {i} ---\n{ch}")

        print("\nFINAL ANSWER:")
        print(answer)
        print("=" * 80)
