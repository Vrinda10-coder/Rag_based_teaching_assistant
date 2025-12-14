# query_rag.py
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity


# 2. Function to embed only the incoming query
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    r.raise_for_status()
    return r.json()["embeddings"]

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response

# Load dataframe from joblib (FAST)
df = joblib.load("chunks_embeddings.joblib")

# 3. Ask a question
incoming_query = input("Ask a Question: ")

# 4. Get embedding for the question (only this API call now)
question_embedding = create_embedding([incoming_query])[0]

# 5. Compute cosine similarities
emb_matrix = np.vstack(df["embedding"].values)   # shape: (num_chunks, dim)
similarities = cosine_similarity(emb_matrix, [question_embedding]).flatten()

# 6. Get top-k most similar chunks
top_k = 5
top_indices = similarities.argsort()[::-1][:top_k]

result_df = df.iloc[top_indices]
# print(result_df[["title", "number", "text" , "start", "end" ]])

# using prompt my model  
prompt = f'''I am teaching Javascript in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{result_df[["title", "number", "start", "end", "text"]].to_json()}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)


with open("response.txt", "w") as f:
    f.write(response)

# for index , item in result_df.iterrows():
#     print(index, item["title"], item["number"], item["text"] , item["start"], item["end"])




