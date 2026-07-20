# 🤖 AI Document Assistant - RAG Chatbot

## 📌 Overview

AI Document Assistant is a Retrieval Augmented Generation (RAG) chatbot that allows users to upload documents and ask questions based on their content.

The application combines document retrieval with Large Language Models (LLMs) to generate context-aware and accurate responses instead of relying solely on the model's internal knowledge.

---

## 🚀 Features

- 📄 Upload PDF and Text documents
- 🔍 Extract text from uploaded files
- ✂️ Split documents into semantic chunks
- 🧠 Generate embeddings using HuggingFace Embeddings
- 🗂️ Store embeddings in Chroma Vector Database
- 🔎 Retrieve the most relevant document chunks using similarity search
- 🤖 Generate answers using Groq Llama 3.3-70B Versatile
- 💬 Interactive Chat Interface using Streamlit
- ⚡ Fast response generation with low latency
- 🔒 Secure API key management using `.env`

---

# 🏗️ Architecture

```
                     User
                       │
                       ▼
              Streamlit Web UI
                       │
                       ▼
              Upload PDF / TXT File
                       │
                       ▼
             Document Loader (LangChain)
                       │
                       ▼
     RecursiveCharacterTextSplitter
          (Chunk Size & Overlap)
                       │
                       ▼
      HuggingFace Embeddings
    (all-MiniLM-L6-v2 - 384 Dimensions)
                       │
                       ▼
         Chroma Vector Database
          (Stores Document Vectors)
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
 User Question              Similarity Search
         │                           │
         └─────────────┬─────────────┘
                       ▼
          Retrieve Relevant Chunks
                       │
                       ▼
            Prompt Construction
      (Context + User Question)
                       │
                       ▼
         Groq Llama 3.3-70B Model
                       │
                       ▼
            AI Generated Answer
                       │
                       ▼
              Display in Streamlit
```

---

# 📂 Project Structure

```
Rag-Chatbot/
│
├── data/
│   ├── ai_notes.txt
│   └── basics of python.pdf
│
├── src/
│   ├── create_embeddings.py
│   ├── create_vectordatabase.py
│   ├── load_document.py
│   ├── rag_chat.py
│   ├── retrieve.py
│   ├── split_text.py
│   └── test.py
│
├── app.py
├── app_upload.py
├── ingest.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Frontend UI |
| LangChain | RAG Pipeline |
| Groq API | Large Language Model |
| Llama 3.3-70B Versatile | Answer Generation |
| HuggingFace Embeddings | Text Embeddings |
| ChromaDB | Vector Database |
| PyPDF | PDF Processing |
| dotenv | Environment Variables |

---

# 🔄 Workflow

1. User uploads PDF or TXT documents.
2. Documents are loaded using LangChain Document Loaders.
3. Documents are split into smaller chunks.
4. Each chunk is converted into embeddings using HuggingFace Embeddings.
5. Embeddings are stored inside Chroma Vector Database.
6. User asks a question.
7. Similarity Search retrieves the most relevant chunks.
8. Retrieved chunks and user query are combined into a prompt.
9. Groq Llama 3.3-70B generates the final answer.
10. The answer is displayed in the Streamlit interface.

---

# ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/DhairyakarshiniNerella/Rag-Chatbot.git
```

## Navigate to the project

```bash
cd Rag-Chatbot
```

## Create virtual environment

```bash
python -m venv venv
```

## Activate virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Create `.env`

```env
GROQ_API_KEY=your_groq_api_key
```

## Create Vector Database

```bash
python ingest.py
```

## Run Application

```bash
streamlit run app.py
```

---





# 👨‍💻 Author

**Dhairyakarshini Nerella**

Computer Science Engineer | AI & Generative AI Enthusiast

GitHub: https://github.com/DhairyakarshiniNerella

---

# ⭐ If you found this project useful, don't forget to star the repository!