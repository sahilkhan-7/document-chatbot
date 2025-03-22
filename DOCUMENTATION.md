# **Documentation for Chat with Documents using RAG**  
This is a Streamlit-based application that allows users to upload and chat with multiple document types (`PDF`, `DOCX`, `TXT`). It uses **LangChain** and **Groq API** to create an LLM-powered conversational chatbot that can retrieve information from uploaded documents and provide structured, concise answers.

---

## 📌 **Overview**
- The app allows users to:
  - Upload multiple document files (PDF, DOCX, TXT)
  - Process the documents with visual progress tracking
  - Store documents in a FAISS vector database
  - Ask questions about the content of the documents
  - Receive structured, accurate answers from Llama 3.3 70B via Groq API
  - Maintain conversational context across multiple interactions
  - Reset the chat history or clear the database when needed

---

## 🚀 **Setup and Configuration**
### **Dependencies**
Install the necessary libraries:
```bash
pip install streamlit langchain groq python-dotenv
pip install langchain-huggingface langchain-community
pip install faiss-cpu
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
from langchain.prompts import ChatPromptTemplate
from groq import Groq
from typing import Optional, List, Mapping, Any
from dotenv import load_dotenv
```
- `streamlit` – For creating the web-based interface
- `langchain` – For building LLM-based retrieval chains
- `groq` – For accessing the Groq API for LLM responses
- `python-dotenv` – For loading environment variables (API key)
- `huggingface` – For creating embeddings

---

### **2. Load Environment Variables**
```python
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```
- Loads the Groq API key from the `.env` file

---

### **3. Define a Custom LLM Using LangChain**
```python
class GroqLLM(LLM):
    """
    Custom LLM class that extends LangChain's base LLM.
    Uses the Groq API to generate responses based on user input and chat history.
    """
    
    # Define class attributes properly
    temperature: float = 0.3
    max_tokens: int = 500
    
    def __init__(self, temperature: float = 0.3, max_tokens: int = 500):
        """Initialize the GroqLLM with temperature and max_tokens parameters."""
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Generates a response from the Groq LLM based on the provided prompt.

        Args:
            prompt (str): The complete prompt with context and question.
            stop (Optional[List[str]]): List of stop tokens (not used here).

        Returns:
            str: The generated response from the LLM.
        """
        client = Groq(api_key=GROQ_API_KEY)
        
        # Define system prompt for better responses
        messages = [
            {"role": "system", "content": 
             """You are a knowledgeable assistant that answers questions based solely on the provided context.
             Focus on giving precise, factual answers that directly address the question.
             Use bullet points for clarity when appropriate.
             If the information isn't in the context, admit you don't know rather than making up answers.
             Format your responses in a clean, readable manner."""}
        ]

        # Append the user prompt (which already contains context from LangChain)
        messages.append({"role": "user", "content": prompt})

        try:
            # Generate a response using Groq LLM
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-specdec",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            response = chat_completion.choices[0].message.content.strip()
            return response
        except Exception as e:
            return f"Error: {e}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Returns the identifying parameters for the custom LLM."""
        return {"name": "GroqLLM", "temperature": self.temperature, "max_tokens": self.max_tokens}

    @property
    def _llm_type(self) -> str:
        """Returns the type of the LLM."""
        return "groq"
```
✅ **Explanation:**  
- Inherits from `LLM` class of LangChain
- Properly initializes with configurable parameters (temperature and max_tokens)
- Uses a custom system prompt to improve response quality
- Uses Llama 3.3 70B model for higher-quality responses
- Implements proper error handling for API interactions
- Returns well-formatted responses optimized for document Q&A

---

### **4. Load Documents with Progress Tracking**
```python
def load_documents(uploaded_files):
    """Loads and processes uploaded files into a list of document objects."""
    documents = []
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Track progress
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
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
        os.remove(temp_filepath)  # Cleanup
        
        # Update progress
        progress_bar.progress((i + 1) / total_files)
    
    return documents
```
✅ **Explanation:**  
- Supports loading multiple file formats
- Visualizes loading progress with a progress bar
- Uses appropriate document loaders based on file extension
- Cleans up temporary files after processing
- Provides visual feedback on the document loading process

---

### **5. Split Documents into Chunks**
```python
def split_documents(documents):
    """Splits the loaded documents into smaller chunks for better context handling."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)
```
✅ **Explanation:**  
- Splits large documents into manageable chunks of 1000 characters
- Uses 200 character overlap to maintain context between chunks
- Specifies custom separators for more natural text splitting
- Optimized chunk size determined through experimentation

---

### **6. Create Vector Store**
```python
def create_vector_store(chunks):
    """Creates a FAISS vector store from document chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)
```
✅ **Explanation:**  
- Uses `all-MiniLM-L6-v2` model for generating high-quality text embeddings
- FAISS stores vector embeddings for efficient similarity search
- Returns a queryable vector store for document retrieval

---

### **7. Create Custom Prompt Template**
```python
def create_custom_prompt():
    """Creates a custom prompt template for better RAG responses."""
    template = """
    Answer the user's question using ONLY the following context and your general knowledge about how to interpret such information.
    If the answer cannot be found in the context, respond with "I don't have enough information to answer that question."
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    YOUR ANSWER:
    """
    return ChatPromptTemplate.from_template(template)
```
✅ **Explanation:**  
- Creates a specialized prompt template for RAG responses
- Explicitly instructs the model to use only provided context
- Includes clear instructions for handling cases where information is missing
- Formats the prompt to clearly separate context from question

---

### **8. Create Enhanced Conversational Retrieval Chain**
```python
def create_conversational_retrieval_chain(vector_store):
    """Creates an enhanced conversational retrieval chain."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Using a custom prompt for more accurate responses
    custom_prompt = create_custom_prompt()
    
    return ConversationalRetrievalChain.from_llm(
        llm=GroqLLM(temperature=0.2, max_tokens=600),
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 4, "fetch_k": 8}
        ),
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        verbose=False
    )
```
✅ **Explanation:**  
- Uses `ConversationBufferMemory` to maintain chat history
- Configures retriever to fetch more candidate documents (fetch_k=8) but return only top matches (k=4)
- Uses a lower temperature (0.2) for more factual responses
- Incorporates custom prompt template for improved answer quality
- Sets higher max_tokens (600) to allow for more comprehensive answers

---

### **9. Handle User Input**
```python
def handle_user_input(user_question):
    """Handles user input and generates a response using the conversation chain."""
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.warning("Please process the documents first.")
        return
    
    with st.spinner("Thinking..."):
        response = st.session_state.conversation({'question': user_question})
    
    # Extract only the answer
    response_text = response.get("answer", "No response generated.")
    
    # Update the chat history for display purposes
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=response_text))
```
✅ **Explanation:**  
- Checks if conversation chain is initialized before processing
- Provides visual feedback during processing with spinner
- Extracts just the answer from the response
- Updates chat history with both user and AI messages
- Provides appropriate error handling and user guidance

---

### **10. Main Function**
```python
def main():
    """
    Main function to set up the Streamlit interface and handle user interactions.
    """
    st.set_page_config(page_title="Document Chat Assistant", layout="wide", page_icon="📚")
    st.header("📚 Chat with Your Documents")
    
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Set up the sidebar for document upload and processing
    with st.sidebar:
        st.subheader("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your files here", 
            type=['pdf', 'docx', 'txt'], 
            accept_multiple_files=True
        )
        
        # Process Files Button
        if st.button("Process Documents", disabled=st.session_state.processing):
            if not uploaded_files:
                st.error("Please upload at least one document.")
            else:
                st.session_state.processing = True
                with st.spinner("Processing documents..."):
                    try:
                        documents = load_documents(uploaded_files)
                        st.write(f"✅ Loaded {len(documents)} document sections")
                        
                        chunks = split_documents(documents)
                        st.write(f"✅ Created {len(chunks)} text chunks")
                        
                        vector_store = create_vector_store(chunks)
                        st.session_state.vector_store = vector_store  # Store vector_store in session state
                        st.write("✅ Created vector store")
                        
                        st.session_state.conversation = create_conversational_retrieval_chain(vector_store)
                        st.success("Documents processed successfully! ✅")
                        st.session_state.processing = False
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.session_state.processing = False
        
        # Clear Chat Button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            if st.session_state.conversation:
                st.session_state.conversation.memory.clear()
            st.success("Chat history cleared! ✅")
        
        # Clear Database Button
        if st.button("Clear Database"):
            st.session_state.vector_store = None
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.success("Vector database cleared! ✅")
        
        # Simple document information
        if uploaded_files:
            st.subheader("📊 Documents")
            for uploaded_file in uploaded_files:
                st.write(f"**{uploaded_file.name}** ({round(uploaded_file.size / 1024, 2)} KB)")
        
        # Database status indicator
        if st.session_state.vector_store is not None:
            st.success("Database: Active ✓")
        else:
            st.warning("Database: Inactive ✗")

    # Chat container with messages
    chat_container = st.container()
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.markdown(f"#### 🧑 **You:**\n{message.content}")
                st.markdown("---")
            elif isinstance(message, AIMessage):
                st.markdown(f"#### 🤖 **Assistant:**\n{message.content}")
                st.markdown("---")
        
        # If no messages, show a simple greeting
        if not st.session_state.chat_history and st.session_state.vector_store is None:
            st.info("Upload your documents in the sidebar and hit 'Process Documents' to get started.")
    
    # Input handling with on_change callback
    user_question = st.text_input(
        "Ask a question about your documents:",
        key="user_input",
        placeholder="Type your question and press Enter...",
        on_change=process_input  # Process on change (Enter key press)
    )
```
✅ **Explanation:**  
- Creates a clean, organized interface with wide layout
- Sets up session state variables for persistent data storage
- Implements document upload and processing in the sidebar
- Provides visual feedback with spinner and success messages
- Includes chat history display with user/assistant formatting
- Uses separate containers for better UI organization
- Shows document information and database status
- Handles text input with automatic processing on Enter key press

---

## 🎯 **Key Improvements**
1. **Progress Tracking**: Added progress bars for visual feedback
2. **Enhanced Error Handling**: Better error detection and user guidance
3. **UI Improvements**: Cleaner chat interface with clear user/assistant distinction
4. **Database Management**: Added ability to clear the vector database
5. **Document Information**: Shows loaded document names and sizes
6. **Status Indicators**: Visual indicators for database status
7. **Custom Prompt Templates**: Specialized prompts for more accurate RAG responses
8. **Optimized Retrieval Parameters**: Fine-tuned k and fetch_k values for better results
9. **Better LLM Integration**: Properly configured GroqLLM with Llama 3.3 70B
10. **Enhanced Chunking Strategy**: Optimized chunk size and overlap

---

## 📊 **Document Processing Flow**
1. Documents are uploaded via the Streamlit interface
2. Each document is processed by the appropriate loader based on file type
3. Documents are split into chunks (1000 chars with 200 overlap)
4. Text chunks are embedded using HuggingFace embeddings
5. Embeddings are stored in a FAISS vector database
6. User queries are processed by the conversational retrieval chain
7. Relevant document chunks are retrieved based on vector similarity
8. LLM generates answers based on retrieved context and the question

---

## 🎯 **How to Run**
```bash
streamlit run app.py
```

---

## 🎮 **Using the Application**
1. Upload documents using the sidebar file uploader
2. Click "Process Documents" to extract and vectorize content
3. Ask questions in the input field at the bottom
4. View assistant responses in the chat container
5. Use "Clear Chat" to reset conversation history
6. Use "Clear Database" to remove all document data and start fresh