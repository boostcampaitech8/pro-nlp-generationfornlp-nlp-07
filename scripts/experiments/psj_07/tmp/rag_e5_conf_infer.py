import os
import ast
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from unsloth import FastLanguageModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


MODEL_NAME = "NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"

TEST_PATH = "data/test/test.csv"
OUTPUT_CSV = "submission_rag_e5_conf.csv"

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 8
SEED = 42

CONF_THRESHOLD = 0.5

PERSIST_DIR = "vectorstores/kowiki_e5_large"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
TOP_K = 4


NUM_MAP = {"①":1,"②":2,"③":3,"④":4,"⑤":5,
           "1":1,"2":2,"3":3,"4":4,"5":5}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model():
    print("[MODEL] loading unsloth model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_retriever():
    print("[RAG] load e5 embeddings + chroma")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    return vs.as_retriever(search_kwargs={"k": TOP_K})


def build_prompt(passage, question, choices, wiki_ctx=""):
    choices_str = "\n".join(f"{i+1}. {c}" for i,c in enumerate(choices))
    wiki_block = wiki_ctx.strip() if wiki_ctx else "없음"

    return f"""너는 한국어 수능형 독해 문제를 푸는 AI 모델이다.
정답은 반드시 1~5 중 하나의 숫자만 출력한다.

[위키 컨텍스트]
{wiki_block}

[지문]
{passage}

[문항]
{question}

[보기]
{choices_str}

정답 번호:"""


def infer_conf(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[:, -1, :]

    tok_ids = [tokenizer(str(i))["input_ids"][-1] for i in ["1","2","3","4","5"]]
    probs = F.softmax(logits[:, tok_ids], dim=-1).squeeze(0)

    best = int(torch.argmax(probs))
    conf = float(probs[best])
    pred = best + 1
    return pred, conf


def retrieve_ctx(retriever, paragraph, question, limit=1500):
    query = f"query: {question}\n\n{paragraph[:200]}"
    docs = retriever.get_relevant_documents(query)

    parts, total = [], 0
    for i,d in enumerate(docs):
        title = d.metadata.get("title","")
        text = d.page_content
        chunk = f"[문서{i+1}] {title}\n{text}"
        if total + len(chunk) > limit: break
        parts.append(chunk); total += len(chunk)

    return "\n\n".join(parts)


def predict_one(model, tokenizer, retriever,
                passage, question, choices):

    # 1️⃣ 먼저 RAG 없이 추론
    prompt = build_prompt(passage, question, choices, wiki_ctx="")
    pred, conf = infer_conf(model, tokenizer, prompt)

    print(f"[BASE] pred={pred}, conf={conf:.3f}")

    # 2️⃣ 확신 낮으면 RAG 켜고 재추론
    if conf < CONF_THRESHOLD:
        wiki_ctx = retrieve_ctx(retriever, passage, question)
        prompt_rag = build_prompt(passage, question, choices, wiki_ctx)
        pred, conf = infer_conf(model, tokenizer, prompt_rag)
        print(f"[RAG USED] new_pred={pred}, conf={conf:.3f}")

    return pred


def main():
    set_seed(SEED)

    model, tokenizer = load_model()
    retriever = load_retriever()

    df = pd.read_csv(TEST_PATH)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    rows = []

    for idx,row in df.iterrows():
        problems = ast.literal_eval(row["problems"])
        passage = row["paragraph"]
        question = problems.get("question","")
        choices = problems.get("choices",[])

        pred = predict_one(
            model, tokenizer, retriever,
            passage, question, choices
        )

        rows.append({"id": row["id"], "answer": int(pred)})

        if (idx+1) % 10 == 0:
            print(f"[PROGRESS] {idx+1}/{len(df)}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {OUTPUT_CSV} done")


if __name__ == "__main__":
    main()
