from __future__ import annotations
"""
Food Lookup: OpenFoodFacts REST API direkt via requests.
Kompatibel mit Python 3.8+.
"""

import requests


class FoodLookup:
    BASE_URL = "https://world.openfoodfacts.net/api/v2/search"

    def search(self, query: str, limit: int = 3) -> list[dict]:
        """Search OpenFoodFacts by product name."""
        try:
            response = requests.get(
                self.BASE_URL,
                params={
                    "categories_tags": query,
                    "search_terms": query,
                    "json": 1,
                    "page_size": limit,
                    "fields": "product_name,brands,nutriments,serving_quantity",
                },
                timeout=5,
            )
            products = response.json().get("products", [])
        except Exception as e:
            print(f"  ⚠️ OpenFoodFacts error: {e}")
            return []

        normalized = []
        for p in products:
            nutriments = p.get("nutriments", {})
            kcal = nutriments.get("energy-kcal_100g") or nutriments.get("energy-kcal")
            if not kcal:
                continue

            normalized.append({
                "name": p.get("product_name", query),
                "brand": p.get("brands", ""),
                "per_100g": {
                    "kcal": float(kcal or 0),
                    "protein_g": float(nutriments.get("proteins_100g", 0) or 0),
                    "carbs_g": float(nutriments.get("carbohydrates_100g", 0) or 0),
                    "fat_g": float(nutriments.get("fat_100g", 0) or 0),
                    "fiber_g": float(nutriments.get("fiber_100g", 0) or 0),
                },
                "serving_size_g": float(p.get("serving_quantity") or 100),
                "source": "openfoodfacts",
            })

        return normalized

    def search_best(self, query: str) -> dict | None:
        """Return top-1 result or None."""
        results = self.search(query, limit=3)
        return results[0] if results else None