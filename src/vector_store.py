from __future__ import annotations
"""
Vector Store: Weaviate-based hybrid search for products.

Hybrid Search = Dense (semantic) + Sparse (BM25 keyword) kombiniert.
Wir liefern Embeddings selbst via sentence-transformers (lokal/GPU).

Collection: "Product"
  - name: str
  - brand: str
  - kcal_per_100g: float
  - protein_per_100g: float
  - carbs_per_100g: float
  - fat_per_100g: float
  - fiber_per_100g: float
  - serving_size_g: float
  - added_by: int (telegram_id)
  - source: str ("ocr_scan" | "manual" | "llm_estimate")
"""

import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer
from src.config import WEAVIATE_HOST, WEAVIATE_PORT, EMBEDDING_MODEL


COLLECTION_NAME = "Product"


class VectorStore:
    def __init__(self):
        print("  🔌 Connecting to Weaviate...")
        self.client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
        )

        print(f"  🤖 Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        self._ensure_collection()
        print("  ✅ VectorStore ready")

    def _ensure_collection(self):
        """Create the Product collection if it doesn't exist yet."""
        if self.client.collections.exists(COLLECTION_NAME):
            print(f"  📦 Collection '{COLLECTION_NAME}' already exists")
            return

        print(f"  📦 Creating collection '{COLLECTION_NAME}'...")
        self.client.collections.create(
            name=COLLECTION_NAME,
            # We supply vectors ourselves → no module needed
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            # BM25 for hybrid search — indexes all text properties automatically
            inverted_index_config=wvc.config.Configure.inverted_index(
                bm25_b=0.75,
                bm25_k1=1.2,
            ),
            properties=[
                wvc.config.Property(
                    name="name",
                    data_type=wvc.config.DataType.TEXT,
                    description="Product name, e.g. 'Gouda Käse Lidl'",
                ),
                wvc.config.Property(
                    name="brand",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="kcal_per_100g",
                    data_type=wvc.config.DataType.NUMBER,
                ),
                wvc.config.Property(
                    name="protein_per_100g",
                    data_type=wvc.config.DataType.NUMBER,
                ),
                wvc.config.Property(
                    name="carbs_per_100g",
                    data_type=wvc.config.DataType.NUMBER,
                ),
                wvc.config.Property(
                    name="fat_per_100g",
                    data_type=wvc.config.DataType.NUMBER,
                ),
                wvc.config.Property(
                    name="fiber_per_100g",
                    data_type=wvc.config.DataType.NUMBER,
                ),
                wvc.config.Property(
                    name="serving_size_g",
                    data_type=wvc.config.DataType.NUMBER,
                ),
                wvc.config.Property(
                    name="added_by",
                    data_type=wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name="source",
                    data_type=wvc.config.DataType.TEXT,
                ),
            ],
        )
        print(f"  ✅ Collection '{COLLECTION_NAME}' created")

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for a text string."""
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def add_product(self, product: dict) -> str:
        """
        Add a product to Weaviate.

        product dict should contain:
            name, brand, per_100g (dict with kcal/protein/carbs/fat/fiber),
            serving_size_g, added_by, source
        """
        per100 = product.get("per_100g", {})

        # Text we embed: name + brand combined for richer semantic matching
        embed_text = product["name"]
        if product.get("brand"):
            embed_text += f" {product['brand']}"

        vector = self._embed(embed_text)

        collection = self.client.collections.get(COLLECTION_NAME)
        uuid = collection.data.insert(
            properties={
                "name": product["name"],
                "brand": product.get("brand", ""),
                "kcal_per_100g": float(per100.get("kcal", 0)),
                "protein_per_100g": float(per100.get("protein_g", 0)),
                "carbs_per_100g": float(per100.get("carbs_g", 0)),
                "fat_per_100g": float(per100.get("fat_g", 0)),
                "fiber_per_100g": float(per100.get("fiber_g", 0)),
                "serving_size_g": float(product.get("serving_size_g", 100)),
                "added_by": int(product.get("added_by", 0)),
                "source": product.get("source", "manual"),
            },
            vector=vector,
        )
        print(f"  🧀 Product added to Weaviate: {product['name']} ({uuid})")
        return str(uuid)

    def search(self, query: str, limit: int = 5,
               alpha: float = 0.5) -> list[dict]:
        """
        Hybrid search: alpha=0.0 → pure BM25, alpha=1.0 → pure semantic.
        Default alpha=0.5 balances both equally.

        Returns list of product dicts with macros + search score.
        """
        query_vector = self._embed(query)
        collection = self.client.collections.get(COLLECTION_NAME)

        results = collection.query.hybrid(
            query=query,           # for BM25
            vector=query_vector,   # for semantic
            alpha=alpha,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(score=True),
        )

        products = []
        for obj in results.objects:
            p = obj.properties
            products.append({
                "name": p["name"],
                "brand": p.get("brand", ""),
                "per_100g": {
                    "kcal": p["kcal_per_100g"],
                    "protein_g": p["protein_per_100g"],
                    "carbs_g": p["carbs_per_100g"],
                    "fat_g": p["fat_per_100g"],
                    "fiber_g": p["fiber_per_100g"],
                },
                "serving_size_g": p.get("serving_size_g", 100),
                "_score": obj.metadata.score,
            })

        return products

    def search_best(self, query: str) -> dict | None:
        """Convenience: return top-1 result or None if collection empty."""
        results = self.search(query, limit=1)
        return results[0] if results else None

    def count(self) -> int:
        """Return number of products in the collection."""
        collection = self.client.collections.get(COLLECTION_NAME)
        response = collection.aggregate.over_all(total_count=True)
        return response.total_count

    def close(self):
        """Close Weaviate connection."""
        self.client.close()