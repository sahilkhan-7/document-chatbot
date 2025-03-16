# 📚 Chat with Multiple Documents using RAG 🚀  

This project is a **Streamlit-based web application** that allows users to upload multiple documents (PDF, DOCX, TXT) and interact with them using a chatbot interface. The chatbot is powered by the **Groq API** and leverages **LangChain** for conversational retrieval, providing structured and concise answers from the uploaded documents.

---

## 🚀 **Project Overview**  
This project integrates **LangChain** and **Groq LLM** to enable users to:  
✅ Upload and process multiple document types (PDF, DOCX, TXT)  
✅ Extract and split document text into manageable chunks  
✅ Store document embeddings using **FAISS** (Facebook AI Similarity Search)  
✅ Retrieve and generate answers based on user queries  
✅ Handle multi-turn conversations with memory retention  

---

## 🏆 **Features**  
✔️ Supports multiple file formats (PDF, DOCX, TXT)  
✔️ Uses **FAISS** for fast and efficient document retrieval  
✔️ Handles conversational context with **LangChain's memory module**  
✔️ Provides structured responses using **Groq LLM**  
✔️ Streamlit-based intuitive UI with file upload and clear chat options  

---

## 🛠️ **Technologies Used**  
| Technology | Purpose |
|-----------|---------|
| **Streamlit** | For building the web-based user interface |
| **LangChain** | For conversational retrieval and memory handling |
| **Groq LLM** | For generating accurate and structured responses |
| **FAISS** | For fast document retrieval using embeddings |
| **HuggingFace Embeddings** | For generating vector embeddings of text chunks |
| **PyPDFLoader, Docx2txtLoader, TextLoader** | For reading and processing documents |

---

## 🌐 **Project Workflow**  

### 1. **Document Upload & Processing**  
- User uploads multiple files (PDF, DOCX, TXT).  
- Files are saved temporarily and processed using appropriate loaders.  
- Text content is extracted and converted into document objects.  

### 2. **Text Splitting & Vectorization**  
- Documents are split into smaller chunks using **RecursiveCharacterTextSplitter**.  
- Text chunks are embedded using **HuggingFace sentence-transformers**.  
- Embeddings are stored in a **FAISS vector store** for efficient retrieval.  

### 3. **Conversational Retrieval Chain**  
- User input is processed using **LangChain's ConversationBufferMemory**.  
- Context from previous turns is added to maintain conversational memory.  
- Groq LLM generates structured and concise responses.  

### 4. **Chat Display & Handling**  
- User input and AI-generated responses are displayed on the interface.  
- "Clear Chat" button resets the session history and memory.  

---

## 🎯 **Impact**  
✅ **Efficient Document-Based Q&A:** The system allows users to extract meaningful insights from large documents quickly.  
✅ **Structured Answers:** Groq LLM ensures that answers are well-organized and concise, enhancing user experience.  
✅ **Fast Retrieval:** FAISS enables fast and scalable search across large datasets.  
✅ **User-Friendly Interface:** The Streamlit-based interface ensures ease of use for both technical and non-technical users.  

---

## 🔮 **Future Scope**  
🚀 **Add Support for More File Formats:** Expand support to CSV, JSON, and other document types.  
🚀 **Advanced Search:** Enable keyword-based search with semantic matching.  
🚀 **Improved Context Handling:** Enhance memory capabilities for long-term conversation handling.  
🚀 **Multi-Model Support:** Allow integration of other LLMs (OpenAI, Claude, DeepSeek) for more diversified responses.  
🚀 **Enhanced UI:** Add options for formatting answers and downloading conversation history.  

---

## 🏆 **Author**  
**Sahil Khan**  
- [LinkedIn](https://linkedin.com/in/sahilkhan7)  
- [GitHub](https://github.com/sahilkhan-7)  

---

## 🏁 **How to Run the Project**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/sahilkhan-7/documents-chatbot.git
cd chat-with-documents
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Create `.env` File**  
Add your Groq API Key in `.env` file:  
```
GROQ_API_KEY="your-groq-api-key"
```

### **4. Run the Application**  
```bash
streamlit run app.py
```