# **Documentation for Chat with Documents using RAG**  
This is a Streamlit-based application that allows users to upload and chat with multiple document types (`PDF`, `DOCX`, `TXT`). It uses **LangChain** and **Groq API** to create an LLM-powered conversational chatbot that can retrieve information from uploaded documents and provide structured, concise answers.

---

## 📌 **Overview**
- The app allows users to:
  - Upload multiple document files (PDF, DOCX, TXT).
  - Process the documents and store them in a vector store using FAISS.
  - Ask questions about the content of the documents.
  - Receive structured, accurate answers from an LLM.
  - Reset the chat history when needed.

---

## 🚀 **Setup and Configuration**
### **Dependencies**
Install the necessary libraries:
```bash
pip install streamlit langchain groq python-dotenv
pip install langchain-huggingface langchain-community
```

### **API Key Setup**
Create a `.env` file in the project directory and add your Groq API key:
```
GROQ_API_KEY="your-api-key"
```

---

## 🏗️ **Code Breakdown**
### **1. Importing Required Libraries**
```python
import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
from typing import Optional, List, Mapping, Any
from dotenv import load_dotenv
```
- `streamlit` – For creating the web-based interface.  
- `langchain` – For building LLM-based retrieval chains.  
- `groq` – For accessing the Groq API for LLM responses.  
- `python-dotenv` – For loading environment variables (API key).  
- `huggingface` – For creating embeddings.  

---

### **2. Load Environment Variables**
```python
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```
- Loads the Groq API key from the `.env` file.

---

### **3. Define a Custom LLM Using LangChain**
```python
class GroqLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = Groq(api_key=GROQ_API_KEY)
        chat_history = st.session_state.get("chat_history", [])
        
        messages = [
            {"role": "system", "content": 
             "You are a helpful assistant. Provide structured and concise responses in bullet points or numbered lists where applicable. Avoid long paragraphs and redundant information."}
        ]
        
        for turn in chat_history:
            if isinstance(turn, HumanMessage):
                messages.append({"role": "user", "content": turn.content})
            elif isinstance(turn, AIMessage):
                messages.append({"role": "assistant", "content": turn.content})

        messages.append({"role": "user", "content": prompt})

        try:
            chat_completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=messages,
                temperature=0.3,
                max_tokens=300  
            )
            response = chat_completion.choices[0].message.content.strip()
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=response))
            return response
        except Exception as e:
            return f"Error: {e}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "GroqLLM"}

    @property
    def _llm_type(self) -> str:
        return "groq"
```
✅ **Explanation:**  
- Inherits from `LLM` class of LangChain.  
- Stores past conversations using `chat_history`.  
- Communicates with Groq API to generate responses.  
- Uses low `temperature` (0.3) for structured and precise output.  
- Limits output length to `300 tokens` for conciseness.  

---

### **4. Load Documents**
```python
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_filepath)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_filepath)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(temp_filepath)
        else:
            loader = UnstructuredFileLoader(temp_filepath)

        documents.extend(loader.load())
        os.remove(temp_filepath)
    return documents
```
✅ **Explanation:**  
- Supports loading multiple file types.  
- Uses LangChain’s document loaders to parse the files.  
- Removes temporary files after processing.  

---

### **5. Split Documents into Chunks**
```python
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks
```
✅ **Explanation:**  
- Splits large documents into manageable chunks.  
- Ensures some overlap between chunks to maintain context.  

---

### **6. Create Vector Store**
```python
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store
```
✅ **Explanation:**  
- Uses `MiniLM` model for generating embeddings.  
- FAISS stores embeddings for quick retrieval.  

---

### **7. Create Conversation Chain**
```python
def create_conversational_retrieval_chain(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=GroqLLM(),
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return qa_chain
```
✅ **Explanation:**  
- Uses LangChain’s `ConversationalRetrievalChain`.  
- Remembers previous interactions using `ConversationBufferMemory`.  

---

### **8. Handle User Input**
```python
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
```
✅ **Explanation:**  
- Takes user input and retrieves relevant context.  
- Generates a response using GroqLLM.  

---

### **9. Main Function**
```python
def main():
    st.set_page_config(page_title="Chat with Multiple Documents", layout="wide", page_icon="💬")
    st.header("Chat with Documents using Groq SDK 🚀")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("Upload your Documents")
        uploaded_files = st.file_uploader("Upload your files", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if uploaded_files:
                documents = load_documents(uploaded_files)
                chunks = split_documents(documents)
                vector_store = create_vector_store(chunks)
                st.session_state.conversation = create_conversational_retrieval_chain(vector_store)
                st.success("Documents processed successfully! ✅")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        if st.session_state.conversation:
            st.session_state.conversation.memory.clear()

    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_user_input(user_question)

    for message in st.session_state.chat_history:
        st.write(f"**{message.content}**")

if __name__ == '__main__':
    main()
```
✅ **Explanation:**  
- Controls overall app flow.  
- Processes uploaded files and sets up the chain.  
- Handles user interaction and displays chat history.  

---

## 🎯 **How to Run**
```bash
streamlit run app.py
```

---

## ✅ **End-to-End Flow:**
1. Upload files  
2. Process files into vector store  
3. Generate answers using GroqLLM  
4. Display structured responses  
5. Clear chat history when needed  