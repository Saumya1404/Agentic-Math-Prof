import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http import models

# ===== CONFIG =====
DATA_PATH = r"Data\train-00000-of-00001.parquet"
QDRANT_PATH = r"Data\knowledge_base\qdrant_db"
COLLECTION_NAME = "gsm8k_knowledge_base"
BATCH_SIZE = 64

# ===== LOAD DATA =====
df = pd.read_parquet(DATA_PATH)

# Combine question and answer into a single content field
df["content"] = df["question"].astype(str) + " " + df["answer"].astype(str)
texts = df["content"].dropna().tolist()

print(f"✅ Loaded {len(texts)} texts from {DATA_PATH}")

# ===== EMBEDDING SETUP =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print(f"🔧 Using device: {device}")

embeddings = []

print("Generating embeddings in batches...")
with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_emb = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        embeddings.extend(batch_emb)
        print(f"Embedded {i + len(batch)} / {len(texts)}")

print("Embeddings generated successfully.")

# ===== QDRANT SETUP =====
client = QdrantClient(path=QDRANT_PATH, prefer_grpc=True)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
)

# ===== INSERT INTO QDRANT =====
UPSERT_BATCH = 500
print("Inserting vectors into Qdrant...")

for i in range(0, len(embeddings), UPSERT_BATCH):
    batch_emb = embeddings[i:i + UPSERT_BATCH]
    batch_text = texts[i:i + UPSERT_BATCH]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=models.Batch(
            ids=list(range(i, i + len(batch_emb))),
            vectors=batch_emb,
            payloads=[{"text": text} for text in batch_text]
        )
    )
    print(f"Inserted {i + len(batch_emb)} / {len(embeddings)}")

print(f"Persistent Qdrant DB '{COLLECTION_NAME}' created successfully at {QDRANT_PATH}")
