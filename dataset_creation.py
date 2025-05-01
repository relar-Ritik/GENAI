import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

with open("CUAD_v1/CUAD_v1.json", "r") as f:
    raw = json.load(f)

examples = []
for doc in tqdm(raw["data"]):
    title = doc["title"]
    for para in doc["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            if qa.get("is_impossible", False):
                continue
            answer = qa["answers"][0]
            examples.append({
                "id": qa["id"],
                "title": title,
                "question": qa["question"],
                "context": context,
                "answer_text": answer["text"],
                "answer_start": answer["answer_start"]
            })

df = pd.DataFrame(examples)
train_df, test_df = train_test_split(df, test_size=0.05)
train_df.to_json("cuad_train_5pct.json", orient="records", lines=True)
test_df.to_json("cuad_test_5pct.json", orient="records", lines=True)
