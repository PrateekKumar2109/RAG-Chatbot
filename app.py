import streamlit as st
import cohere
import faiss
import numpy as np
import PyPDF2
from io import BytesIO
import pickle
import tempfile
import os
from typing import List, Dict, Tuple
import re

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Cohere client
@st.cache_resource
def init_cohere():
    """Initialize Cohere client with API key"""
    return cohere.Client("YPcYZ15gTNg2O8Hssqq4cyHBiUcwgpUN9uxaQ59y")

co = init_cohere()

class DocumentProcessor:
    """Class to handle PDF processing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks

class EmbeddingManager:
    """Class to handle embeddings and FAISS operations"""
    
    def __init__(self):
        self.index = None
        self.chunks = []
        self.embeddings = []
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using Cohere"""
        try:
            with st.spinner("Creating embeddings..."):
                # Process in batches to avoid API limits
                batch_size = 96  # Cohere's batch limit
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = co.embed(
                        texts=batch,
                        model="embed-english-v3.0",
                        input_type="search_document"
                    )
                    all_embeddings.extend(response.embeddings)
                
                return np.array(all_embeddings, dtype=np.float32)
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return np.array([])
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings"""
        try:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            
            return index
        except Exception as e:
            st.error(f"Error building FAISS index: {str(e)}")
            return None
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar chunks using the query"""
        try:
            # Create query embedding
            query_response = co.embed(
                texts=[query],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            query_embedding = np.array(query_response.embeddings, dtype=np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
            
            return results
        except Exception as e:
            st.error(f"Error searching similar chunks: {str(e)}")
            return []

class RAGSystem:
    """Main RAG system class"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.document_processor = DocumentProcessor()
    
    def process_document(self, pdf_file) -> bool:
        """Process uploaded PDF and create embeddings"""
        try:
            # Extract text from PDF
            text = self.document_processor.extract_text_from_pdf(pdf_file)
            if not text:
                return False
            
            # Clean and preprocess text
            text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
            text = text.strip()
            
            # Chunk the text
            chunks = self.document_processor.chunk_text(text)
            if not chunks:
                st.error("No text chunks created from the document")
                return False
            
            # Create embeddings
            embeddings = self.embedding_manager.create_embeddings(chunks)
            if embeddings.size == 0:
                return False
            
            # Build FAISS index
            index = self.embedding_manager.build_faiss_index(embeddings)
            if index is None:
                return False
            
            # Store in embedding manager
            self.embedding_manager.index = index
            self.embedding_manager.chunks = chunks
            self.embedding_manager.embeddings = embeddings
            
            return True
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return False
    
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate answer using Cohere's generation model"""
        try:
            # Prepare context
            context = "\n\n".join(context_chunks)
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say "I cannot find the answer in the provided context."

Context:
{context}

Question: {question}

Answer:"""
            
            # Generate response
            response = co.generate(
                model="command",
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
                stop_sequences=["Question:", "Context:"]
            )
            
            return response.generations[0].text.strip()
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return "Sorry, I encountered an error while generating the answer."
    
    def answer_question(self, question: str) -> Dict:
        """Main method to answer questions"""
        if self.embedding_manager.index is None:
            return {
                "answer": "Please upload and process a PDF document first.",
                "sources": []
            }
        
        # Search for relevant chunks
        similar_chunks = self.embedding_manager.search_similar_chunks(question, k=5)
        
        if not similar_chunks:
            return {
                "answer": "No relevant information found in the document.",
                "sources": []
            }
        
        # Extract chunks and scores
        context_chunks = [chunk for chunk, score in similar_chunks]
        
        # Generate answer
        answer = self.generate_answer(question, context_chunks)
        
        return {
            "answer": answer,
            "sources": similar_chunks
        }

# Initialize RAG system
@st.cache_resource
def init_rag_system():
    return RAGSystem()

def main():
    """Main Streamlit application"""
    st.title("📚 RAG-based Q&A Assistant")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # Initialize RAG system
    rag_system = init_rag_system()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    success = rag_system.process_document(uploaded_file)
                    
                if success:
                    st.success("Document processed successfully!")
                    st.session_state.document_processed = True
                else:
                    st.error("Failed to process document")
                    st.session_state.document_processed = False
        
        # Display processing status
        if hasattr(st.session_state, 'document_processed') and st.session_state.document_processed:
            st.info("✅ Document ready for questions")
            
            # Show document stats
            if rag_system.embedding_manager.chunks:
                st.metric("Text Chunks", len(rag_system.embedding_manager.chunks))
                st.metric("Embeddings", len(rag_system.embedding_manager.embeddings))
    
    # Main area for Q&A
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🤔 Ask Questions")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What is this document about?",
            help="Ask any question about the uploaded document"
        )
        
        if st.button("Get Answer", type="primary") and question:
            with st.spinner("Searching for answer..."):
                result = rag_system.answer_question(question)
            
            # Display answer
            st.subheader("Answer:")
            st.write(result["answer"])
            
            # Display sources
            if result["sources"]:
                with st.expander("📋 Source Context (click to expand)"):
                    for i, (chunk, score) in enumerate(result["sources"], 1):
                        st.markdown(f"**Source {i}** (Similarity: {score:.3f})")
                        st.text_area(f"Context {i}", chunk, height=100, key=f"source_{i}")
    
    with col2:
        st.header("ℹ️ Instructions")
        st.markdown("""
        1. **Upload** a PDF document using the sidebar
        2. **Process** the document by clicking the button
        3. **Ask** questions about the document content
        4. **Review** the answer and source context
        
        ### Features:
        - 🔍 Semantic search using Cohere embeddings
        - 📊 FAISS vector database for fast retrieval
        - 🤖 Cohere's language model for answer generation
        - 📝 Source context for transparency
        """)
        
        st.header("🛠️ Technical Details")
        st.markdown("""
        - **Embeddings**: Cohere embed-english-v3.0
        - **LLM**: Cohere Command model
        - **Vector DB**: FAISS (Facebook AI Similarity Search)
        - **Chunking**: 500 words with 50 word overlap
        """)

if __name__ == "__main__":
    main()
