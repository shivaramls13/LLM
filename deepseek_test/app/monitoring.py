# app/monitoring.py
#import time
import logging
from prometheus_client import Counter, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_app")

# Prometheus metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total number of queries')
RETRIEVAL_TIME = Histogram('rag_retrieval_time_seconds', 'Time spent retrieving documents')
GENERATION_TIME = Histogram('rag_generation_time_seconds', 'Time spent generating responses')
RESPONSE_LENGTH = Histogram('rag_response_length_chars', 'Length of generated responses in characters')

def start_monitoring_server(port=9000):
    """Start a Prometheus metrics server"""
    start_http_server(port)
    logger.info(f"Monitoring server started on port {port}")

def log_retrieval_performance(query, docs):
    """Log document retrieval performance"""
    QUERY_COUNTER.inc()
    logger.info(f"Retrieved {len(docs)} documents for query: {query}")

def log_generation_performance(query, generation_time, response_length):
    """Log response generation performance"""
    GENERATION_TIME.observe(generation_time)
    RESPONSE_LENGTH.observe(response_length)
    logger.info(f"Generated response in {generation_time:.2f}s for query: {query}")
