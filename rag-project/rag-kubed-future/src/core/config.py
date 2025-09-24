import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")
    VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")
    MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = 384  # Adjust based on the model used

config = Config()