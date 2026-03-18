import os
from dotenv import load_dotenv

load_dotenv()

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = "llava:7b"
TEXT_MODEL = "llama3.1:8b"

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "health_tracker")

# Qdrant (Sprint 2)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Energy tracking
GPU_TDP_WATTS = float(os.getenv("GPU_TDP_WATTS", "220"))
ENERGY_COST_PER_KWH = float(os.getenv("ENERGY_COST_PER_KWH", "0.28"))
