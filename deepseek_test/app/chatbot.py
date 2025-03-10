# app/chatbot.py
from app.retriever import DocumentRetriever
from app.generator import ResponseGenerator

class Chatbot:
    def __init__(self, document_path, model_name="deepseek-r1:7b"):
        self.retriever = DocumentRetriever(document_path)
        self.generator = ResponseGenerator(model_name)
        self.conversation_history = []
        self.vectorstore = None
        
    def initialize(self):
        self.vectorstore = self.retriever.process_documents()
        
    def chat(self, query, temperature=0.5, include_thinking=False):
        if not self.vectorstore:
            self.initialize()
        
        # Get relevant documents
        docs = self.retriever.retrieve_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        result = self.generator.generate_response(query, context, temperature)
        
        # Store in conversation history
        self.conversation_history.append({
            "query": query,
            "response": result["answer"]
        })
        
        if include_thinking:
            return result
        else:
            return result["answer"]
