import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L5-v2'
)

client = chromadb.Client()

collection_name = "my_grocery_collection"

def main():
    try:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "A collection for storing grocery data"},
            configuration={
                "hnsw": {"space", "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")
    except Exception as error:  # Catch any errors and log them to the console
        print(f"Error: {error}")

texts = [
    'fresh red apples',
    'organic bananas',
    'ripe mangoes',
    'whole wheat bread',
    'farm-fresh eggs',
    'natural yogurt',
    'frozen vegetables',
    'grass-fed beef',
    'free-range chicken',
    'fresh salmon fillet',
    'aromatic coffee beans',
    'pure honey',
    'golden apple',
    'red fruit'
]

ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]