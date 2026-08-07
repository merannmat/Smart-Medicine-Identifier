import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("Loading dataset...")
df = pd.read_csv('dataset/Medicine_Details.csv')

# Fill empty values
df['Composition'] = df['Composition'].fillna('')
df['Uses'] = df['Uses'].fillna('')
df['Medicine Name'] = df['Medicine Name'].fillna('')

# Combine text for embedding
df['combined'] = (
    df['Medicine Name'] + ' ' +
    df['Composition'] + ' ' +
    df['Uses']
)

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Creating embeddings... (2-3 min lagenge)")
embeddings = model.encode(
    df['combined'].tolist(),
    show_progress_bar=True,
    batch_size=64
)

# Normalize for cosine similarity
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)

# FAISS index with Inner Product (cosine after normalize)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index and dataframe
faiss.write_index(index, 'utils/medicine_index.faiss')
df.to_pickle('utils/medicine_data.pkl')

print("✅ Done! Index saved.")
print(f"Total medicines indexed: {len(df)}")
