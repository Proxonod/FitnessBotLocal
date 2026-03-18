from __future__ import annotations
"""
Database: MongoDB operations for the Health Tracker.

Collections:
  - users:    Per-user profile (telegram_id, name, goals)
  - meals:    Daily meal log entries
  - products: Learned product database (scanned labels)
  - recipes:  Auto-generated cookbook
  - usage:    Token/energy tracking per call
"""

from datetime import datetime, timezone
from pymongo import MongoClient
from src.config import MONGO_URI, MONGO_DB


class Database:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[MONGO_DB]

        # Collections
        self.users = self.db["users"]
        self.meals = self.db["meals"]
        self.products = self.db["products"]
        self.recipes = self.db["recipes"]
        self.usage = self.db["usage"]

        # Indexes
        self.users.create_index("telegram_id", unique=True)
        self.meals.create_index([("telegram_id", 1), ("date", 1)])
        self.products.create_index("name")
        self.usage.create_index("timestamp")

    # --- Users ---

    def get_or_create_user(self, telegram_id: int, name: str) -> dict:
        """Get user profile or create new one."""
        user = self.users.find_one({"telegram_id": telegram_id})
        if user is None:
            user = {
                "telegram_id": telegram_id,
                "name": name,
                "created_at": datetime.now(timezone.utc),
                "goals": {
                    "daily_kcal": None,  # User sets later
                    "daily_protein_g": None,
                },
                "preferences": {},
            }
            self.users.insert_one(user)
            print(f"  👤 New user created: {name} ({telegram_id})")
        return user

    def update_user_goals(self, telegram_id: int, goals: dict):
        """Update user's nutrition goals."""
        self.users.update_one(
            {"telegram_id": telegram_id},
            {"$set": {"goals": goals}}
        )

    # --- Meals ---

    def log_meal(self, telegram_id: int, meal_data: dict) -> str:
        """
        Log a meal entry.

        meal_data should contain:
            description: str - what was eaten
            items: list[dict] - individual items with macros
            total: dict - total kcal, protein, carbs, fat, fiber
            portion: float - portion multiplier (1.0 = full, 0.5 = half)
            photo_id: str | None - Telegram photo file_id
        """
        entry = {
            "telegram_id": telegram_id,
            "timestamp": datetime.now(timezone.utc),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            **meal_data,
        }
        result = self.meals.insert_one(entry)
        return str(result.inserted_id)

    def get_daily_meals(self, telegram_id: int,
                        date: str | None = None) -> list[dict]:
        """Get all meals for a user on a given date."""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return list(self.meals.find(
            {"telegram_id": telegram_id, "date": date},
            {"_id": 0}
        ))

    def get_daily_totals(self, telegram_id: int,
                         date: str | None = None) -> dict:
        """Aggregate daily nutrition totals."""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        pipeline = [
            {"$match": {"telegram_id": telegram_id, "date": date}},
            {"$group": {
                "_id": None,
                "total_kcal": {"$sum": "$total.kcal"},
                "total_protein": {"$sum": "$total.protein_g"},
                "total_carbs": {"$sum": "$total.carbs_g"},
                "total_fat": {"$sum": "$total.fat_g"},
                "total_fiber": {"$sum": "$total.fiber_g"},
                "meal_count": {"$sum": 1},
            }}
        ]
        result = list(self.meals.aggregate(pipeline))
        if result:
            return result[0]
        return {
            "total_kcal": 0, "total_protein": 0, "total_carbs": 0,
            "total_fat": 0, "total_fiber": 0, "meal_count": 0,
        }

    # --- Products (learned from label scans) ---

    def save_product(self, product_data: dict) -> str:
        """
        Save a product to the learned database.

        product_data should contain:
            name: str - product name (e.g. "Gouda Käse Lidl")
            brand: str | None
            per_100g: dict - kcal, protein, carbs, fat, fiber per 100g
            serving_size_g: float | None
            added_by: int - telegram_id who scanned it
            source: str - "ocr_scan" or "manual"
        """
        product_data["created_at"] = datetime.now(timezone.utc)
        result = self.products.insert_one(product_data)
        print(f"  🧀 Product saved: {product_data['name']}")
        return str(result.inserted_id)

    def find_product(self, name: str) -> dict | None:
        """Simple text search for a product. Sprint 2 adds vector search."""
        return self.products.find_one(
            {"name": {"$regex": name, "$options": "i"}},
            {"_id": 0}
        )

    def list_products(self, telegram_id: int | None = None) -> list[dict]:
        """List all known products, optionally filtered by who added them."""
        query = {"added_by": telegram_id} if telegram_id else {}
        return list(self.products.find(query, {"_id": 0}))

    # --- Usage tracking ---

    def log_usage(self, telegram_id: int, stats_dict: dict):
        """Log a model call for token/energy tracking."""
        entry = {
            "telegram_id": telegram_id,
            "timestamp": datetime.now(timezone.utc),
            **stats_dict,
        }
        self.usage.insert_one(entry)

    def get_usage_summary(self, telegram_id: int | None = None) -> dict:
        """Get accumulated usage stats."""
        match = {"telegram_id": telegram_id} if telegram_id else {}
        pipeline = [
            {"$match": match},
            {"$group": {
                "_id": None,
                "total_prompt_tokens": {"$sum": "$prompt_tokens"},
                "total_completion_tokens": {"$sum": "$completion_tokens"},
                "total_energy_kwh": {"$sum": "$energy_kwh"},
                "total_cost_eur": {"$sum": "$energy_cost_eur"},
                "total_api_equiv_usd": {"$sum": "$estimated_api_cost_usd"},
                "call_count": {"$sum": 1},
            }}
        ]
        result = list(self.usage.aggregate(pipeline))
        return result[0] if result else {}
