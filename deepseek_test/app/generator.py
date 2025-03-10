# app/generator.py
import ollama
import time
from app.monitoring import log_generation_performance

class ResponseGenerator:
    def __init__(self, model_name="deepseek-r1:7b"):
        self.model_name = model_name
    
    def generate_response(self, query, context, temperature=0.5, extract_thinking=True):
        start_time = time.time()
        
        prompt = f"""Answer the following question based only on the provided context.
        If you cannot answer from the context, say "I don't have enough information."
        Think step by step before answering.
        
        Context:
        {context}
        
        Question: {query}
        
        <think>
        Think through the reasoning process step by step here. Explore the context and determine the best answer.
        </think>
        
        Answer:"""
        
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": temperature}
        )
        
        generation_time = time.time() - start_time
        
        # Extract thinking and answer
        full_response = response["response"]
        
        # Parse thinking section
        thinking = ""
        if "<think>" in full_response and "</think>" in full_response:
            thinking_start = full_response.find("<think>") + len("<think>")
            thinking_end = full_response.find("</think>")
            thinking = full_response[thinking_start:thinking_end].strip()
        
        # Get answer (everything after the </think> tag)
        answer = ""
        if "</think>" in full_response:
            answer = full_response[full_response.find("</think>") + len("</think>"):].strip()
        else:
            answer = full_response  # No thinking tags found, use full response
        
        # Log generation performance
        log_generation_performance(query, generation_time, len(full_response))
        
        if extract_thinking:
            return {"thinking": thinking, "answer": answer}
        else:
            return answer
