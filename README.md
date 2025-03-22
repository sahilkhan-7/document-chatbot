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
✔️ Provides structured responses using **Groq LLM API** with Llama 3.3 70B  
✔️ Custom prompt templates for better RAG responses  
✔️ Progress tracking for document processing  
✔️ Streamlit-based intuitive UI with file upload and database management  

---

## 🛠️ **Technologies Used**  
| Technology | Purpose |
|-----------|---------|
| **Streamlit** | For building the web-based user interface |
| **LangChain** | For conversational retrieval and memory handling |
| **Groq API** | For generating accurate and structured responses using Llama 3.3 70B |
| **FAISS** | For fast document retrieval using embeddings |
| **HuggingFace Embeddings** | For generating vector embeddings of text chunks |
| **PyPDFLoader, Docx2txtLoader, TextLoader** | For reading and processing documents |

---

## 🌐 **Project Workflow**  

### 1. **Document Upload & Processing**  
- User uploads multiple files (PDF, DOCX, TXT).  
- Files are saved temporarily and processed using appropriate loaders.  
- Text content is extracted and converted into document objects.  
- Progress bar tracks document processing status.

### 2. **Text Splitting & Vectorization**  
- Documents are split into smaller chunks using **RecursiveCharacterTextSplitter**.  
- Text chunks are embedded using **HuggingFace sentence-transformers**.  
- Embeddings are stored in a **FAISS vector store** for efficient retrieval.  

### 3. **Conversational Retrieval Chain**  
- User input is processed using **LangChain's ConversationBufferMemory**.  
- Context from previous turns is added to maintain conversational memory.  
- Custom GroqLLM class integrates with Groq API to generate structured responses.  
- Custom prompt template enhances the quality of RAG responses.

### 4. **Chat Display & Handling**  
- User input and AI-generated responses are displayed in a chat interface.  
- "Clear Chat" button resets the session history and memory.  
- "Clear Database" button removes the vector store and resets the application.

---

## 🎯 **Impact**  
✅ **Efficient Document-Based Q&A:** The system allows users to extract meaningful insights from large documents quickly.  
✅ **Structured Answers:** The Llama 3.3 70B model ensures that answers are well-organized and concise, enhancing user experience.  
✅ **Fast Retrieval:** FAISS enables fast and scalable search across large datasets.  
✅ **User-Friendly Interface:** The Streamlit-based interface ensures ease of use for both technical and non-technical users.  

---

## 💪 **Challenges Faced & Solutions**

### 1. **Optimal Chunk Size Configuration**
- **Challenge:** Finding the ideal chunk size and overlap for document splitting.
- **Solution:** After experimentation, settled on a chunk size of 1000 characters with 200-character overlap for optimal context retention and retrieval precision.

### 2. **Custom GroqLLM Integration**
- **Challenge:** Implementing a custom LangChain LLM class to interface with the Groq API.
- **Solution:** Developed a custom `GroqLLM` class that extends LangChain's base LLM, properly handling the API requests, message formatting, and error handling for seamless integration.

### 3. **Model Parameter Optimization**
- **Challenge:** Correctly configuring and passing model parameters like temperature and max_tokens.
- **Solution:** Implemented proper parameter handling in the GroqLLM class constructor and _call method, ensuring these parameters are correctly applied to API requests.

### 4. **Prompt Engineering**
- **Challenge:** Creating effective prompts to guide the model's responses.
- **Solution:** Developed a custom system prompt and template that instructs the model to focus on the provided context and format answers clearly, improving response relevance and quality.

### 5. **Conversation Memory Management**
- **Challenge:** Maintaining coherent conversation history and context across multiple interactions.
- **Solution:** Correctly implemented ConversationBufferMemory with appropriate memory keys and return message settings to preserve conversation context.

### 6. **Retrieval Chain Configuration**
- **Challenge:** Setting up an effective retrieval chain with appropriate parameters.
- **Solution:** After multiple iterations, configured the ConversationalRetrievalChain with optimized search parameters (k=4, fetch_k=8) and custom prompt templates.

### 7. **Multi-Format Document Processing**
- **Challenge:** Handling various document formats consistently.
- **Solution:** Implemented format-specific loaders (PyPDFLoader, Docx2txtLoader, TextLoader) with appropriate fallback to UnstructuredFileLoader for other formats.

### 8. **UI/UX Design**
- **Challenge:** Creating an intuitive interface for non-technical users.
- **Solution:** Developed a clean Streamlit interface with progress indicators, status messages, and separate panels for document management and chat interaction.

---

## 🔮 **Future Scope**  
🚀 **Add Support for More File Formats:** Expand support to CSV, JSON, and other document types.  
🚀 **Advanced Search:** Enable keyword-based search with semantic matching.  
🚀 **Improved Context Handling:** Enhance memory capabilities for long-term conversation handling.  
🚀 **Multi-Model Support:** Allow integration of other LLMs (OpenAI, Claude, DeepSeek) for more diversified responses.  
🚀 **Enhanced UI:** Add options for formatting answers and downloading conversation history.  
🚀 **Document Preprocessing:** Add options for text extraction quality improvement and OCR for scanned documents.

---

## 🏆 **Author**  
**Sahil Khan**  
- [LinkedIn](https://linkedin.com/in/sahilkhan7)  
- [GitHub](https://github.com/sahilkhan-7)  

---

## 🏁 **How to Run the Project**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/sahilkhan-7/document-chatbot.git
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

### **5. Using the App**
1. Upload documents using the sidebar file uploader
2. Click "Process Documents" to extract and vectorize the content
3. Ask questions in the input field at the bottom
4. Use "Clear Chat" to reset the conversation or "Clear Database" to remove all documents