import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2'
)

client = chromadb.Client()

collection_name = "my_grocery_collection"

def main():
    try:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"description": "A collection for storing grocery data","hnsw:space": "cosine"},
        )
        print(f"Collection created: {collection.name}")
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

        collection.add(
            documents=texts,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
            ids=ids
        )
        all_items = collection.get()
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")
        perform_similarity_search(collection,all_items)
    except Exception as error:  # Catch any errors and log them to the console
        print(f"Error: {error}")

def perform_similarity_search(collection, all_items):
    try:
        query_term = "apple"
        results = collection.query(
            query_texts=[query_term],
            n_results=3
        )
        print(f"Query results for '{query_term}':")
        print(results)
    except Exception as error:
        print(f"Error in similarity search: {error}")

if __name__ == "__main__":
    main()