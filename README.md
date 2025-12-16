# 🤖 RAG PDF Chatbot using Gemini API

A **Retrieval-Augmented Generation (RAG)** based chatbot that allows users to upload a PDF document and interactively ask questions strictly based on the document’s content.  
The system uses **Google Gemini models for embeddings and generation**, combined with semantic similarity search.<img width="1440" height="900" alt="Screenshot 2025-12-16 at 7 23 12 PM" src="https://github.com/user-attachments/assets/d0f11ab7-e678-4c63-ab0b-84e543ce2b1f" />


---

## 📌 Project Overview

This project demonstrates how **Large Language Models (LLMs)** can be combined with **document retrieval techniques** to build a reliable, document-aware chatbot.

The chatbot:
- Extracts text from uploaded PDFs
- Splits content into overlapping chunks
- Converts chunks into embeddings
- Retrieves the most relevant chunks using cosine similarity
- Generates answers **only from the uploaded document**

This ensures **zero hallucination** and **strict document grounding**.

---

## 🧠 Architecture (RAG Pipeline)

PDF Upload

↓

Text Extraction

↓

Chunking (with overlap)

↓

Gemini Embeddings

↓

Vector Similarity Search

↓

Context Injection

↓

Gemini Response Generation



---

## ✨ Features

- 📄 Upload and process PDF documents
- 🔍 Semantic search using embeddings
- 🤖 Gemini-powered question answering
- 📊 Document statistics (pages, chunks, characters)
- 💬 Interactive Streamlit chat UI
- 🧠 Strict document-based answering (no external knowledge)
- 🧾 Chat history tracking
- 🖥 Terminal logging for debugging (question, chunks, answer)

---

## 🛠 Tech Stack

| Category | Technology |
|--------|------------|
| Frontend | Streamlit |
| LLM | Google Gemini |
| Embeddings | Gemini Embedding Model |
| Similarity Search | Cosine Similarity (scikit-learn) |
| PDF Processing | PyPDF |
| Backend | Python |
| Environment Management | python-dotenv |

---

## 📂 Project Structure

rag-pdf-chatbot/
│

├── chatbot.py # Main Streamlit application

├── requirements.txt # Project dependencies

├── .gitignore # Ignored files (.env, venv, etc.)

├── README.md # Project documentation

└── .env # API key (NOT pushed to Git)



---

## 🔑 Environment Setup

1️⃣ Clone Repository

git clone https://gitlab.com/your-username/rag-pdf-chatbot.git
cd rag-pdf-chatbot

2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Environment Variable
Create a .env file:
GEMINI_API_KEY=your_api_key_here

▶️ Running the Application
streamlit run <file name>
The app will open in your browser.


🧪 How to Use

1. Open the application
2. Navigate to Process PDF
3. Upload a PDF document
4. Click Process Now
5. Go to Chat Interface
6. Ask questions related to the uploaded document

If a question is outside the document scope, the chatbot will respond professionally:

"I’m unable to answer this question as it falls outside the scope of the uploaded document."


📊 Debugging Output (Terminal)

For mentor review and debugging, the following are printed in terminal:

* User question
* Retrieved document chunks
* Final generated answer

This improves traceability and transparency.


🚀 Future Enhancements

* Vector database integration (FAISS / ChromaDB)
* Support for multiple documents
* Conversation memory across sessions
* File type support beyond PDF
* Cloud deployment

👨‍💻 Author

Vasuki A

vasukiarul27@gmail.com

AI / ML Engineer Intern

https://www.linkedin.com/posts/vasuki27_ai-generativeai-rag-activity-7406694458991439873-4oHl?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFWofHABP5vZ1q4SVksdeQ_qxpl9ilnOKXM


📜 License

This project is created for learning and internship evaluation purposes.

---
