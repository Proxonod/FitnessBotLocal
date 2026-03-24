"""
Health Tracker Bot — Main Entry Point

Start with:
    python main.py

Requirements:
    1. Ollama running: ollama serve
    2. Models pulled: ollama pull llava:7b && ollama pull llama3.1:8b
    3. Docker containers: docker compose up -d
    4. .env file with TELEGRAM_BOT_TOKEN
"""

from src.bot import create_bot


def main():
    print("🚀 Starting Health Tracker Bot...")
    print("   Ctrl+C to stop\n")

    bot = create_bot()
    bot.run_polling(drop_pending_updates=False) # False when offline MSG should get read also


if __name__ == "__main__":
    main()
