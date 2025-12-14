import json
import requests
import os
import pandas as pd



def create_embeddings(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    r.raise_for_status()
    data = r.json()
    return data["embeddings"]

jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    # list of texts for this file
    texts = [c["text"] for c in content["chunks"]]
    embeddings = create_embeddings(texts)

    if len(embeddings) != len(content["chunks"]):
        raise ValueError(f"Embeddings: {len(embeddings)}, chunks: {len(content['chunks'])}")

    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

        
with open("all_chunks_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(my_dicts, f, ensure_ascii=False)







    







