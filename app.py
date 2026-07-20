from dotenv import load_dotenv
import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ------------------------
# Load Environment
# ------------------------

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")


# ------------------------
# Page Configuration
# ------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------
# Initialize Memory
# ------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []



# ------------------------
# Sidebar UI
# ------------------------

with st.sidebar:

    st.title("🤖 AI Assistant")


    st.markdown(
    """
    ---
    ### 🚀 Features

    ✅ PDF Question Answering  
    ✅ Groq AI Responses  
    ✅ Semantic Search  
    ✅ Chroma Vector Database  
    ✅ RAG Pipeline  
    ✅ Chat Memory

    ---
    """
    )


    st.subheader("⚙️ Model Information")


    st.success(
    """
    ⚡ LLM

    Groq Llama 3.3 70B
    """
    )


    st.info(
    """
    🔎 Embedding

    HuggingFace MiniLM
    """
    )


    st.warning(
    """
    🗄️ Database

    Chroma DB
    """
    )


    st.divider()


    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()



# ------------------------
# Main UI
# ------------------------

st.markdown(
"""
## 📚 Chat with your PDF using RAG


Ask questions and get AI-powered answers using:

- ⚡ Groq Llama Model
- 🔍 Chroma Vector Search
- 🔗 LangChain
- 🧠 Chat Memory

"""
)



# ------------------------
# Display Previous Chat
# ------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])



# ------------------------
# Load Embedding Model
# ------------------------

embeddings = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)



# ------------------------
# Load Chroma Database
# ------------------------

vector_db = Chroma(

    persist_directory="vectorstore",

    embedding_function=embeddings

)



# ------------------------
# Load Groq LLM
# ------------------------

llm = ChatGroq(

    model="llama-3.3-70b-versatile",

    temperature=0.3,

    groq_api_key=groq_key

)



# ------------------------
# Chat Input
# ------------------------

query = st.chat_input(
    "Ask something..."
)



if query:


    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    with st.chat_message("user"):

        st.markdown(query)



    # ------------------------
    # Create History
    # ------------------------

    chat_history = ""


    for message in st.session_state.messages[:-1]:

        chat_history += f"""

{message['role']}:

{message['content']}

"""



    greetings = [

        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "thanks",
        "thank you",
        "bye"

    ]



    # ------------------------
    # Greeting
    # ------------------------

    if query.lower().strip() in greetings:


        prompt = f"""

You are a friendly AI assistant.

Reply naturally and shortly.

Previous Conversation:

{chat_history}


User:

{query}


Answer:

"""


    else:


        # ------------------------
        # Retrieve Documents
        # ------------------------

        docs = vector_db.similarity_search(

            query,

            k=3

        )



        context = "\n\n".join(

            doc.page_content

            for doc in docs

        )



        # ------------------------
        # RAG + Memory Prompt
        # ------------------------

        prompt = f"""

You are an AI document assistant.

Answer the user's question using the context.

Use previous conversation if required.

If the answer is not available in the document,
say you don't find it in the document but provide
general knowledge if useful.


Previous Conversation:

{chat_history}


Document Context:

{context}


Current Question:

{query}


Answer:

"""



    # ------------------------
    # Generate Response
    # ------------------------

    with st.spinner("Thinking..."):


        response = llm.invoke(prompt)


        answer = response.content



    # ------------------------
    # Display Assistant
    # ------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)



    # Save assistant message

    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": answer
        }

    )