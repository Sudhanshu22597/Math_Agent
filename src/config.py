import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model configuration
GEMINI_MODEL_NAME = "gemini-2.0-flash" # Or another suitable Gemini model
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Or another suitable embedding model

# Vector Store configuration
VECTOR_STORE_PATH = "faiss_index_jee_math"
CSV_PATH = "data/jee_math.csv"

# Agent configuration
# Increased threshold - FAISS L2 distance, lower is better.
# Higher value here means we accept less similar results from KB.
SIMILARITY_THRESHOLD = 1.0 # Increased from 0.75, adjust further if needed (e.g., 1.2)
MAX_WEB_RESULTS = 3

# Guardrails configuration
ALLOWED_TOPICS = ["math", "mathematics", "algebra", "geometry", "calculus", "statistics", "probability", "education"]
PRIVACY_KEYWORDS = ["password", "secret", "credit card", "social security"] # Add more sensitive keywords
