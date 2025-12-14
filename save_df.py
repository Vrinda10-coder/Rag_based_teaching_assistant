# save_df.py
import json
import pandas as pd
import joblib

# Load chunks + embeddings (already generated earlier)
with open("all_chunks_with_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df[["chunk_id", "start", "end", "title", "number", "text", "embedding"]]

# Save dataframe using joblib
joblib.dump(df, "chunks_embeddings.joblib")

print("DataFrame saved to chunks_embeddings.joblib")
print("Shape:", df.shape)
