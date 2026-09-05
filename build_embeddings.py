import json
import re
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def generate_embeddings(texts, tokenizer, session):
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(length=None)
    encoded_list = tokenizer.encode_batch(texts)
    
    input_ids = np.array([e.ids for e in encoded_list], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encoded_list], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encoded_list], dtype=np.int64)

    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })
    token_embeddings = outputs[0]

    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(mask_expanded, token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    embeddings = sum_embeddings / sum_mask

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / np.clip(norms, a_min=1e-9, a_max=None)).astype(np.float32)

def main():
    with open("qa_database.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        for key in ["faqs", "questions", "data", "qa_pairs"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    rows = []
    for p in data:
        if isinstance(p, dict) and "question" in p:
            rows.append(p["question"])
            for alt in p.get("alt_questions", []):
                rows.append(alt)

    tokenizer_file = hf_hub_download(repo_id=MODEL_NAME, filename="tokenizer.json")
    model_path = hf_hub_download(repo_id=MODEL_NAME, filename="onnx/model.onnx")
    
    tokenizer = Tokenizer.from_file(tokenizer_file)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    embeddings = generate_embeddings(rows, tokenizer, session)
    np.save("qa_embeddings.npy", embeddings)
    print(f"Saved {embeddings.shape} embeddings to qa_embeddings.npy successfully.")

if __name__ == "__main__":
    main()