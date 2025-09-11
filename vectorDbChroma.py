import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ------------------------------
# 1. Load dataset
# ------------------------------
csv_path = "orders.csv"
orders_df = pd.read_csv(csv_path)

# Clean column names
def clean_columns(df):
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df

orders_df = clean_columns(orders_df)

print("Sample data:")
print(orders_df.head())

# ------------------------------
# 2. Convert rows to text + metadata
# ------------------------------
def row_to_text(row, table="orders"):
    """Convert dataframe row into descriptive string for embeddings."""
    return f"Table: {table}. " + ", ".join([f"{k}: {v}" for k, v in row.items()])

# text version (for semantic embedding)
order_texts = [row_to_text(row) for row in orders_df.to_dict(orient="records")]

# structured metadata (filterable form)
order_metadata = orders_df.assign(table="orders").to_dict(orient="records")

# ------------------------------
# 3. Create embeddings
# ------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # light model, ~100MB
order_embeddings = embedder.encode(order_texts, convert_to_numpy=True).tolist()

# ------------------------------
# 4. ChromaDB setup
# ------------------------------
client = chromadb.PersistentClient(path="./chroma_orders_db")
client.delete_collection(name="orders_collection")
collection = client.create_collection(name="orders_collection")
collection = client.get_or_create_collection(name="orders_collection")

# Insert data into vector DB
collection.add(
    documents=order_texts,
    metadatas=order_metadata,
    embeddings=order_embeddings,
    ids=[f"order_{i}" for i in range(len(order_texts))]
)

print(f"✅ Inserted {len(order_texts)} rows into ChromaDB.")

# ------------------------------
# 5. Example query
# ------------------------------
query = "Give me the aggregate of the freight that was shipped to Aachen."
query_embedding = embedder.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

print("\nTop results:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print("Doc:", doc)
    print("Meta:", meta, "\n")
