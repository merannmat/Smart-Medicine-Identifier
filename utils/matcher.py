import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz

print("Loading RAG model...")

# -----------------------------
# Load model, FAISS index and dataset
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("utils/medicine_index.faiss")

df = pd.read_pickle("utils/medicine_data.pkl")

# Cache medicine names for faster searching
medicine_names = (
    df["Medicine Name"]
    .fillna("")
    .str.lower()
    .tolist()
)


def find_medicine(query_text):

    if not query_text:
        return None

    query_text = query_text.lower().strip()

    # =====================================
    # STEP 1 : RapidFuzz Search
    # =====================================

    match = process.extractOne(
        query_text,
        medicine_names,
        scorer=fuzz.ratio
    )

    if match:

        matched_name = match[0]
        score = match[1]

        print("\n===== RAPIDFUZZ =====")
        print("Best Match :", matched_name)
        print("Score      :", score)

        if score >= 90:

            row = df[
                df["Medicine Name"].str.lower() == matched_name
            ].iloc[0]

            return {
                "medicine_name": row["Medicine Name"],
                "composition": row["Composition"],
                "uses": row["Uses"],
                "side_effects": row.get("Side_effects", "N/A"),
                "manufacturer": row.get("Manufacturer", "N/A"),
                "confidence": int(score)
            }

    # =====================================
    # STEP 2 : FAISS Search
    # =====================================

    print("\nRapidFuzz confidence too low.")
    print("Trying FAISS semantic search...")

    embedding = model.encode([query_text])
    embedding = np.array(embedding).astype("float32")

    faiss.normalize_L2(embedding)

    distances, indices = index.search(embedding, k=1)

    best_idx = indices[0][0]

    if best_idx == -1:
        return None

    similarity = float(distances[0][0])

    print("\n===== FAISS =====")
    print("Similarity :", similarity)

    if similarity < 0.70:
        return None

    row = df.iloc[best_idx]

    return {
        "medicine_name": row["Medicine Name"],
        "composition": row["Composition"],
        "uses": row["Uses"],
        "side_effects": row.get("Side_effects", "N/A"),
        "manufacturer": row.get("Manufacturer", "N/A"),
        "confidence": int(similarity * 100)
    }