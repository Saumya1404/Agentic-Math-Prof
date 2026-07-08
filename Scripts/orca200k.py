import pandas as pd
import torch
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http import models

# ===== CONFIG =====
DATA_PATH = r"Data\orca_train-00000-of-00001 (1).parquet"
QDRANT_PATH = r"Data\knowledge_base\qdrant_db_orca_sample"
COLLECTION_NAME = "orca_200k_sample"
BATCH_SIZE = 64  # For embedding
UPSERT_BATCH = 500  # For inserting into Qdrant
SAMPLE_FRACTION = 0.1  # 10% of dataset
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# ===================


def check_gpu():
    """Checks GPU availability."""
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("No GPU detected. Falling back to CPU — expect slower embedding.")
    return "cpu"


def main():
    start_time = time.time()
    device = check_gpu()

    # ===== LOAD DATA =====
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} total rows.")

    # Sample 10%
    df = df.sample(frac=SAMPLE_FRACTION, random_state=42)
    print(f"Using {len(df)} documents (~10% of total).")

    # Combine question and answer
    df["content"] = df["question"].astype(str) + "\n" + df["answer"].astype(str)
    texts = df["content"].dropna().tolist()

    print(f"Prepared {len(texts)} combined question-answer texts.")

    # ===== EMBEDDING MODEL =====
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"Loaded embedding model: {EMBEDDING_MODEL}")

    embeddings = []
    print("Generating embeddings in batches...")
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_emb = model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings.extend(batch_emb)
            print(f"Embedded {i + len(batch)} / {len(texts)}")

    print(f"Generated {len(embeddings)} embeddings.")

    # ===== QDRANT SETUP =====
    print(f"Creating Qdrant collection '{COLLECTION_NAME}' at {QDRANT_PATH}")
    client = QdrantClient(path=QDRANT_PATH, prefer_grpc=True)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=len(embeddings[0]),
            distance=Distance.COSINE
        ),
    )

    # ===== INSERT INTO QDRANT =====
    print("Inserting embeddings into Qdrant in batches...")
    for i in range(0, len(embeddings), UPSERT_BATCH):
        batch_emb = embeddings[i:i + UPSERT_BATCH]
        batch_text = texts[i:i + UPSERT_BATCH]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=models.Batch(
                ids=list(range(i, i + len(batch_emb))),
                vectors=batch_emb,
                payloads=[{"text": text} for text in batch_text],
            ),
        )
        print(f"Inserted {i + len(batch_emb)} / {len(embeddings)}")

    print(f"Successfully stored {len(embeddings)} entries in Qdrant.")

    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Qdrant DB Path: {QDRANT_PATH}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
