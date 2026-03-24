from src.vector_store import VectorStore

vs = VectorStore()
print('Products in DB:', vs.count())

# Test: ein Produkt einfügen
vs.add_product({
    'name': 'Gouda Käse',
    'brand': 'Lidl',
    'per_100g': {'kcal': 356, 'protein_g': 25, 'carbs_g': 0.5, 'fat_g': 27, 'fiber_g': 0},
    'serving_size_g': 30,
    'added_by': 0,
    'source': 'manual'
})

# Test hybrid search
results = vs.search('Käse', alpha=0.5)
print('Search results:', results)
vs.close()