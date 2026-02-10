# rag_utils.py
import re
import torch
import numpy as np

def filter_docs(docs, question, topn=5):
    q_tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", question))
    scored = []
    for d in docs:
        text = d["text"]
        s_tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", text))
        overlap = len(q_tokens & s_tokens)
        if overlap > 0:
            scored.append((overlap, text))
    scored.sort(reverse=True)
    return [t for _, t in scored[:topn]]

@torch.inference_mode()
def run_rag(model, tokenizer, base_prompt, docs, allow_labels, max_len):
    ctx = "\n\n".join([f"[문서]\n{d}" for d in docs])
    rag_prompt = base_prompt.replace(
        "정답:",
        f"\n\n[참고 문서]\n{ctx}\n\n정답:"
    )

    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        rag_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(model.device)

    out = model(**inputs)
    scores = out.logits[0, -1]

    label_scores = []
    for l in allow_labels:
        ids = tokenizer.encode(str(l), add_special_tokens=False)
        label_scores.append(scores[ids[-1]].item())

    probs = torch.softmax(torch.tensor(label_scores), dim=-1)
    margin = float(torch.topk(probs, 2).values[0] - torch.topk(probs, 2).values[1])
    ent = float((-probs * torch.log(probs + 1e-12)).sum())
    top1p = float(probs.max())

    pred = str(allow_labels[int(torch.argmax(probs))])
    return pred, margin, ent, top1p, rag_prompt
