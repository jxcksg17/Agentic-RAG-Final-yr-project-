import json
import re
import string
import requests
from typing import List, Dict

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

def normalize_text(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common) / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

def call_phi4_mini(prompt: str) -> str:
    try:
        payload = {
            "model": "microsoft/Phi-4-mini-instruct",
            "messages": [
                {"role": "system", "content": "Answer the question directly and concisely."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 128
        }
        response = requests.post(VLLM_API_URL, json=payload, timeout=5).json()
        return response["choices"][0]["message"]["content"].strip()
    except Exception:
        return "mocked answer"

def mock_single_step_retrieval(query: str) -> str:
    return "Retrieved single-step contextual evidence relevant to the question."

def mock_multi_step_retrieval(query: str) -> str:
    return "Retrieved multi-hop evidence: Step 1 facts merged with Step 2 relational facts."

def simulate_and_label(queries_data: List[Dict], f1_threshold: float = 0.5) -> List[Dict]:
    labeled_data = []
    for item in queries_data:
        query = item["query"]
        ground_truth = item["ground_truth"]

        # Strategy 0: No Retrieval[span_1](start_span)[span_1](end_span)
        ans_0 = call_phi4_mini(f"Question: {query}\nAnswer:")
        if compute_f1(ans_0, ground_truth) >= f1_threshold:
            labeled_data.append({"query": query, "label": 0})
            continue

        # Strategy 1: Single-Step Retrieval[span_2](start_span)[span_2](end_span)
        ctx_1 = mock_single_step_retrieval(query)
        ans_1 = call_phi4_mini(f"Context: {ctx_1}\n\nQuestion: {query}\nAnswer:")
        if compute_f1(ans_1, ground_truth) >= f1_threshold:
            labeled_data.append({"query": query, "label": 1})
            continue

        # Strategy 2: Multi-Step Retrieval[span_3](start_span)[span_3](end_span)
        labeled_data.append({"query": query, "label": 2})

    return labeled_data

if __name__ == "__main__":
    sample_queries = [
        {"query": "What is the capital of France?", "ground_truth": "Paris"},
        {"query": "Who directed the movie Inception?", "ground_truth": "Christopher Nolan"}
    ]
    results = simulate_and_label(sample_queries)
    with open("engineer_2/configs/sample_labeled_queries.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Labeling test complete. Saved to engineer_2/configs/sample_labeled_queries.json")