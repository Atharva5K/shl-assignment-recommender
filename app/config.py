"""
Configuration module for the SHL Assessment Recommender.
Centralizes all configuration values.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CATALOG_PATH = BASE_DIR / "shl_product_catalogue.json"

# FAISS
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "metadata.pkl"

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# FAISS retrieval
FAISS_TOP_K = 30  # Retrieve more candidates for reranking
FINAL_TOP_K = 10  # Final recommendations cap

# Retrieval confidence thresholds
SIMILARITY_THRESHOLD = 0.25  # Minimum cosine similarity for useful results
CONFIDENCE_HIGH = 0.45  # High-confidence threshold

# Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MAX_TOKENS = 1024
GROQ_TEMPERATURE = 0.1  # Low temperature for deterministic, grounded responses

# Conversation constraints
MAX_TURNS = 8  # Evaluator caps at 8 turns

# Assessment categories (from catalog)
ASSESSMENT_CATEGORIES = [
    "Ability & Aptitude",
    "Assessment Exercises",
    "Biodata & Situational Judgment",
    "Competencies",
    "Development & 360",
    "Knowledge & Skills",
    "Personality & Behavior",
    "Simulations",
]

# Job levels (from catalog)
JOB_LEVELS = [
    "Director",
    "Entry-Level",
    "Executive",
    "Front Line Manager",
    "General Population",
    "Graduate",
    "Manager",
    "Mid-Professional",
    "Professional Individual Contributor",
    "Supervisor",
]
