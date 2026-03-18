# 🥗 Health Tracker Agent

Local AI-powered nutrition tracker via Telegram. All models run locally on GPU — no API keys, no cloud, full privacy.

## Stack
- **LLaVA 7B** — Food recognition + label OCR (via Ollama)
- **Llama 3.1 8B** — Reasoning + meal parsing (via Ollama)  
- **MongoDB** — User profiles, meal logs, product database
- **Qdrant** — Vector search for products (Sprint 2)
- **Telegram Bot API** — User interface

## Quick Start

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Start Ollama (if not running)
ollama serve &

# 3. Install Python deps
pip install -r requirements.txt --break-system-packages

# 4. Verify setup
python test_setup.py

# 5. Run the bot
python main.py
```

## Commands
- `/start` — Welcome + help
- `/today` — Daily nutrition summary
- `/stats` — Token + energy usage
- `/products` — Your product database

## Architecture
See project conversation for full architecture diagrams.
