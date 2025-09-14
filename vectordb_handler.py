import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


class VectorDBHandler:
    def __init__(self, db_path: str = "./chroma_db", embed_model: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model and ChromaDB client."""
        self.embedder = SentenceTransformer(embed_model)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collections = {}

    @staticmethod
    def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names: lowercase, replace spaces with underscores."""
        df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
        return df

    @staticmethod
    def row_to_text(row: dict, table: str) -> str:
        """Convert a row dict to descriptive text for embeddings."""
        return f"Table: {table}. " + ", ".join([f"{k}: {v}" for k, v in row.items()])

    def prepare_documents(self, df: pd.DataFrame, table_name: str):
        """Generate text and metadata from dataframe rows."""
        records = df.to_dict(orient="records")
        texts = [self.row_to_text(r, table_name) for r in records]
        metadata = df.assign(table=table_name).to_dict(orient="records")
        return texts, metadata

    def create_collection(self, name: str, overwrite: bool = True):
        """Create or reset a collection."""
        if overwrite:
            try:
                self.client.delete_collection(name=name)
            except Exception:
                pass
        self.collections[name] = self.client.get_or_create_collection(name=name)
        return self.collections[name]

    def insert_dataframe(self, df: pd.DataFrame, collection_name: str, table_name: str):
        """Insert dataframe into a ChromaDB collection."""
        df = self.clean_columns(df)
        texts, metadata = self.prepare_documents(df, table_name)
        embeddings = self.embedder.encode(texts, convert_to_numpy=True).tolist()

        collection = self.create_collection(collection_name)
        collection.add(
            documents=texts,
            metadatas=metadata,
            embeddings=embeddings,
            ids=[f"{table_name}_{i}" for i in range(len(texts))]
        )

        print(f"✅ Inserted {len(texts)} rows into collection '{collection_name}'.")
        return collection

    def query(self, collection_name: str, user_query: str, n_results: int = 5):
        """Run a semantic query against a collection."""
        if collection_name not in self.collections:
            self.collections[collection_name] = self.client.get_or_create_collection(collection_name)

        query_embedding = self.embedder.encode([user_query]).tolist()
        results = self.collections[collection_name].query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results

    def clear_collection(self, collection_name: str):
        """Delete a specific collection and its data."""
        try:
            self.client.delete_collection(name=collection_name)
            self.collections.pop(collection_name, None)
            print(f"🗑️ Cleared collection '{collection_name}'.")
        except Exception as e:
            print(f"⚠️ Failed to clear collection '{collection_name}': {e}")

    def clear_all_collections(self):
        """Delete all collections from the database."""
        try:
            all_collections = self.client.list_collections()
            for col in all_collections:
                self.client.delete_collection(name=col.name)
            self.collections.clear()
            print("🗑️ All collections cleared from the database.")
        except Exception as e:
            print(f"⚠️ Failed to clear all collections: {e}")
