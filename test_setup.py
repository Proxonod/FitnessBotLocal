"""
Quick smoke test — run this before the bot to verify everything works.

    python test_setup.py
"""

import sys


def test_ollama():
    """Test Ollama connection and models."""
    print("1️⃣  Testing Ollama connection...")
    try:
        import ollama
        models = ollama.list()
        model_names = [m.model for m in models.models]
        print(f"   ✅ Connected. Models available: {model_names}")

        if "llava:7b" not in model_names:
            print("   ⚠️  llava:7b not found! Run: ollama pull llava:7b")
            return False
        if "llama3.1:8b" not in model_names:
            print("   ⚠️  llama3.1:8b not found! Run: ollama pull llama3.1:8b")
            return False

        # Quick inference test with text model
        print("   Testing Llama 3.1 inference...")
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "Sag nur 'OK' — nichts anderes."}],
        )
        tokens = response.get("eval_count", 0)
        print(f"   ✅ Llama 3.1 works! Response: {response['message']['content'].strip()[:50]}")
        print(f"   📊 Tokens generated: {tokens}")
        return True

    except Exception as e:
        print(f"   ❌ Ollama failed: {e}")
        print("   → Is 'ollama serve' running?")
        return False


def test_mongodb():
    """Test MongoDB connection."""
    print("\n2️⃣  Testing MongoDB connection...")
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        print("   ✅ MongoDB connected")

        # Quick write/read test
        db = client["health_tracker_test"]
        db["test"].insert_one({"test": True})
        db["test"].drop()
        print("   ✅ Read/write works")
        client.close()
        return True

    except Exception as e:
        print(f"   ❌ MongoDB failed: {e}")
        print("   → Run: docker compose up -d")
        return False


def test_telegram_token():
    """Check that the bot token is set."""
    print("\n3️⃣  Checking Telegram token...")
    try:
        from src.config import TELEGRAM_BOT_TOKEN
        if TELEGRAM_BOT_TOKEN and len(TELEGRAM_BOT_TOKEN) > 20:
            print(f"   ✅ Token loaded ({TELEGRAM_BOT_TOKEN[:10]}...)")
            return True
        else:
            print("   ❌ Token missing or too short")
            print("   → Check .env file")
            return False
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Health Tracker — Setup Test\n")

    results = [
        test_ollama(),
        test_mongodb(),
        test_telegram_token(),
    ]

    print("\n" + "=" * 40)
    if all(results):
        print("✅ Alles bereit! Starte den Bot mit: python main.py")
    else:
        print("⚠️  Einige Tests fehlgeschlagen — siehe oben.")
        sys.exit(1)
