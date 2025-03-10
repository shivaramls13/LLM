# app/retriever.py
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from app.embeddings import get_embeddings
from app.monitoring import log_retrieval_performance

class DocumentRetriever:
    def __init__(self, document_path, chunk_size=1000, chunk_overlap=200):
        self.document_path = document_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        
    def process_documents(self):
        # Load documents
        if not os.path.exists(self.document_path):
            raise FileNotFoundError(f"Document not found: {self.document_path}")
        
        documents = TextLoader(self.document_path).load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vectorstore
        embeddings = get_embeddings()
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return self.vectorstore
    
    def retrieve_documents(self, query, k=3):
        if not self.vectorstore:
            raise ValueError("Documents have not been processed yet")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        
        # Log retrieval performance
        log_retrieval_performance(query, docs)
        
        return docs
