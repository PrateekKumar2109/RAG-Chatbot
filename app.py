import streamlit as st
import cohere
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
import re
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Cohere client
COHERE_API_KEY = "vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg"
co = cohere.Client(COHERE_API_KEY)

class DocumentProcessor:
    """Handle PDF processing and text extraction"""
    
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
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks - reduced size for faster processing"""
        if not text.strip():
            return []
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit total text length to prevent timeouts
        max_chars = 50000  # Limit to 50k characters
        if len(text) > max_chars:
            text = text[:max_chars]
            st.warning(f"Document truncated to {max_chars} characters to prevent timeout.")
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings in the last 100 characters
                last_period = text.rfind('.', start, end)
                last_question = text.rfind('?', start, end)
                last_exclamation = text.rfind('!', start, end)
                
                sentence_end = max(last_period, last_question, last_exclamation)
                
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks

class EmbeddingManager:
    """Manage embeddings using Cohere"""
    
    def __init__(self):
        self.embeddings_cache = {}
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts using Cohere with batching"""
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]
            
            if not valid_texts:
                return np.array([])
            
            # Process in smaller batches to avoid timeouts
            batch_size = 10  # Reduce batch size for faster processing
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]
                
                # Show progress
                progress = (i + len(batch)) / len(valid_texts)
                st.progress(progress, f"Processing embeddings: {i + len(batch)}/{len(valid_texts)}")
                
                response = co.embed(
                    texts=batch,
                    model='embed-english-v3.0',
                    input_type='search_document'
                )
                
                all_embeddings.extend(response.embeddings)
            
            return np.array(all_embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            st.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a single query"""
        try:
            response = co.embed(
                texts=[query],
                model='embed-english-v3.0',
                input_type='search_query'
            )
            return np.array(response.embeddings[0])
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            st.error(f"Error generating query embedding: {e}")
            return np.array([])

class RAGSystem:
    """RAG system combining retrieval and generation"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.document_chunks = []
        self.chunk_embeddings = None
    
    def process_document(self, pdf_file) -> bool:
        """Process uploaded PDF and create embeddings with timeout protection"""
        try:
            # Extract text from PDF
            with st.spinner("Extracting text from PDF..."):
                text = DocumentProcessor.extract_text_from_pdf(pdf_file)
            
            if not text.strip():
                st.error("No text could be extracted from the PDF. Please check if the PDF contains readable text.")
                return False
            
            st.success(f"✅ Extracted {len(text)} characters from PDF")
            
            # Chunk the text
            with st.spinner("Creating text chunks..."):
                self.document_chunks = DocumentProcessor.chunk_text(text)
            
            if not self.document_chunks:
                st.error("No valid text chunks could be created from the document.")
                return False
            
            st.info(f"📄 Created {len(self.document_chunks)} text chunks")
            
            # Limit number of chunks to prevent timeout
            max_chunks = 50  # Limit to 50 chunks for faster processing
            if len(self.document_chunks) > max_chunks:
                self.document_chunks = self.document_chunks[:max_chunks]
                st.warning(f"⚠️ Limited to first {max_chunks} chunks to prevent timeout")
            
            # Generate embeddings for chunks
            st.info("🔄 Generating embeddings (this may take a moment)...")
            self.chunk_embeddings = self.embedding_manager.get_embeddings(self.document_chunks)
            
            if self.chunk_embeddings.size == 0:
                st.error("Failed to generate embeddings for the document.")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            st.error(f"Error processing document: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        if self.chunk_embeddings is None or len(self.document_chunks) == 0:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding_manager.get_query_embedding(query)
            
            if query_embedding.size == 0:
                return []
            
            # Calculate similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), 
                self.chunk_embeddings
            )[0]
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            relevant_chunks = []
            for idx in top_indices:
                relevant_chunks.append({
                    'text': self.document_chunks[idx],
                    'similarity': similarities[idx],
                    'index': idx
                })
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            return []
    
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Generate answer using Cohere's generation model"""
        try:
            if not relevant_chunks:
                return "I couldn't find relevant information in the document to answer your question."
            
            # Prepare context from relevant chunks
            context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
            
            # Create prompt
            prompt = f"""Based on the following context from the document, please answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer: """
            
            # Generate response using Cohere
            response = co.generate(
                model='command-r-plus',
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
                stop_sequences=["Question:", "Context:"]
            )
            
            answer = response.generations[0].text.strip()
            
            # Add source information
            answer += f"\n\n*Based on {len(relevant_chunks)} relevant sections from the uploaded document.*"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {e}"

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    return RAGSystem()

def main():
    st.set_page_config(
        page_title="PDF Q&A with Cohere RAG",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Question Answering with Cohere RAG")
    st.markdown("Upload a PDF document and ask questions about its content using Cohere's powerful language models.")
    
    # Initialize RAG system
    rag_system = get_rag_system()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
            
            # Show file info
            file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
            st.caption(f"File size: {file_size:.1f} MB")
            
            if file_size > 10:
                st.warning("⚠️ Large files may take longer to process and might timeout.")
            
            # Process document button
            if st.button("🔄 Process Document", type="primary"):
                success = rag_system.process_document(uploaded_file)
                if success:
                    st.success("✅ Document processed successfully!")
                    st.session_state.document_processed = True
                    st.balloons()
                else:
                    st.error("❌ Failed to process document")
                    st.session_state.document_processed = False
        
        # Document info
        if hasattr(st.session_state, 'document_processed') and st.session_state.document_processed:
            st.info(f"📄 Document ready for questions!\n\nChunks: {len(rag_system.document_chunks)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Check if document is processed
        if not hasattr(st.session_state, 'document_processed') or not st.session_state.document_processed:
            st.warning("👈 Please upload and process a PDF document first using the sidebar.")
            return
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="What is this document about?",
            key="question_input"
        )
        
        # Search button
        if st.button("🔍 Search", type="primary", disabled=not query.strip()):
            if query.strip():
                with st.spinner("Searching for relevant information..."):
                    # Retrieve relevant chunks
                    relevant_chunks = rag_system.retrieve_relevant_chunks(query, top_k=3)
                    
                    if relevant_chunks:
                        # Generate answer
                        with st.spinner("Generating answer..."):
                            answer = rag_system.generate_answer(query, relevant_chunks)
                        
                        # Display answer
                        st.subheader("📋 Answer")
                        st.write(answer)
                        
                        # Store in session state for reference
                        if 'qa_history' not in st.session_state:
                            st.session_state.qa_history = []
                        
                        st.session_state.qa_history.append({
                            'question': query,
                            'answer': answer,
                            'relevant_chunks': relevant_chunks
                        })
                    else:
                        st.error("No relevant information found for your question.")
        
        # Display Q&A history
        if hasattr(st.session_state, 'qa_history') and st.session_state.qa_history:
            st.subheader("📝 Previous Questions")
            for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5 Q&As
                with st.expander(f"Q: {qa['question'][:50]}..."):
                    st.write("**Question:**", qa['question'])
                    st.write("**Answer:**", qa['answer'])
    
    with col2:
        st.header("ℹ️ How it works")
        st.markdown("""
        1. **Upload PDF**: Choose a PDF document
        2. **Process**: Click 'Process Document' to extract and chunk text
        3. **Embeddings**: Create vector embeddings using Cohere
        4. **Ask**: Enter your question
        5. **Retrieve**: Find most relevant text chunks
        6. **Generate**: Get AI-powered answers using Cohere's language model
        """)
        
        # Show relevant chunks if available
        if (hasattr(st.session_state, 'qa_history') and 
            st.session_state.qa_history and 
            st.session_state.qa_history[-1].get('relevant_chunks')):
            
            st.subheader("🎯 Source Chunks")
            st.caption("Most relevant sections used for the last answer:")
            
            for i, chunk in enumerate(st.session_state.qa_history[-1]['relevant_chunks'][:2]):
                with st.expander(f"Chunk {i+1} (Similarity: {chunk['similarity']:.3f})"):
                    st.text(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])

if __name__ == "__main__":
    main()
