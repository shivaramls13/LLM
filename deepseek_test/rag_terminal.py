# terminal_rag.py

import os
from pathlib import Path
import time
from app.chatbot import Chatbot

def main():
    # Initialize chatbot
    script_dir = Path(__file__).resolve().parent
    document_path = script_dir / "data" / "manual.txt"
    
    print("Initializing RAG chatbot...")
    chatbot = Chatbot(document_path)
    chatbot.initialize()
    print("RAG chatbot initialized and ready. Type 'exit' to quit.")
    
    # Main chat loop
    while True:
        # Get user input
        user_query = input("\nYou: ")
        
        # Check if user wants to exit
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Exiting RAG chatbot. Goodbye!")
            break
        
        # Process query
        start_time = time.time()
        try:
            # Setting default values as requested
            temperature = 0.4
            include_thinking = False
            
            # Process the query
            result = chatbot.chat(
                user_query,
                temperature=temperature,
                include_thinking=include_thinking
            )
            
            processing_time = time.time() - start_time
            
            # Display result
            print("\nDeepseek:")
            print(result)
            print(f"\n[Processed in {processing_time:.2f} seconds]")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
