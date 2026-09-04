import os
import re
import csv
import json
import gc
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
torch.set_num_threads(1)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed filename -- NOT auto-detected. Auto-detecting "whichever .json file
# is first in the folder" is fragile: if more than one .json file exists
# (backups, old versions, etc.) it can silently load the wrong database.
QA_DB_PATH = os.path.join(BASE_DIR, "qa_database.json")

EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "qa_embeddings.npy")
EMBEDDINGS_CACHE_META_PATH = os.path.join(BASE_DIR, "qa_embeddings_meta.json")
UNANSWERED_LOG_PATH = os.path.join(BASE_DIR, "unanswered_questions_log.csv")
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, "feedback_log.csv")

MODEL_NAME = "all-MiniLM-L6-v2"

# Tuned and tested against real advisor rephrasings earlier in this project.
SIMILARITY_THRESHOLD = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.68

STAGE_MAPPING = {
    "T1": ["T1", "TEACH"],
    "T2": ["T2", "TEST"],
    "T3": ["T3", "TREAT"],
    "T4": ["T4", "TRACK"],
}

START_PHRASES = {
    "how to start", "how do i start", "getting started", "get started",
    "how to begin", "how do i begin", "where to start", "how to use this",
    "what can you do", "help", "start", "hi", "hello", "hey", "greetings",
}

START_ANSWER = (
    "Welcome to myFinergy! You can ask me questions about advisor workflows, "
    "FinFit diagnostics, client conversations, or the T1-T4 framework. "
    "Try asking something like: 'How do I pitch FinFit to a client?' or "
    "'What is the Finergy Score?'"
)

FALLBACK_ANSWER = (
    "I don't have a confident answer for that yet in the myFinergy knowledge base. "
    "Could you rephrase the question, or check with your team lead? "
    "I've logged this question so it can be added to the database."
)


class ChatRequest(BaseModel):
    question: str
    stage: Optional[str] = None  # optional hint from the frontend; not required


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str


_qa_pairs: list[dict] = []
_qa_embeddings: np.ndarray | None = None
_exact_match_index: dict[str, dict] = {}
_qa_rows: list[tuple[str, dict]] = []
_model: SentenceTransformer | None = None


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _load_qa_database() -> list[dict]:
    """
    Supports both a flat list [ {...}, {...} ] and a wrapped object
    { "faqs": [ {...}, {...} ] }.
    """
    with open(QA_DB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ["faqs", "questions", "data", "qa_pairs"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    return [item for item in data if isinstance(item, dict) and "question" in item]


def _expand_with_alt_questions(qa_pairs: list[dict]) -> list[tuple[str, dict]]:
    rows = []
    for p in qa_pairs:
        rows.append((p["question"], p))
        for alt in p.get("alt_questions", []):
            rows.append((alt, p))
    return rows


def _generate_embeddings(texts: list[str]) -> np.ndarray:
    global _model
    with torch.no_grad():
        embeddings = _model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=16,
        )
    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    gc.collect()
    return embeddings.astype(np.float32)


def _build_or_load_embeddings(rows: list[tuple[str, dict]]) -> np.ndarray:
    texts = [text for text, _ in rows]
    fingerprint = {
        "count": len(texts),
        "texts_hash": hash(tuple(texts)),
        "model": MODEL_NAME,
    }

    if os.path.exists(EMBEDDINGS_CACHE_PATH) and os.path.exists(EMBEDDINGS_CACHE_META_PATH):
        try:
            with open(EMBEDDINGS_CACHE_META_PATH, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if cached_meta == fingerprint:
                return np.load(EMBEDDINGS_CACHE_PATH)
        except Exception:
            pass

    embeddings = _generate_embeddings(texts)

    np.save(EMBEDDINGS_CACHE_PATH, embeddings)
    with open(EMBEDDINGS_CACHE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f)

    return embeddings


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _qa_pairs, _qa_embeddings, _exact_match_index, _qa_rows, _model

    # NOTE: deliberately NOT forcing HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE here.
    # Those only work if the model is already cached locally; forcing them
    # unconditionally breaks the very first run on a fresh machine. If you
    # want fully offline operation after the first successful run, set
    # HF_HUB_OFFLINE=1 as an actual environment variable when launching
    # uvicorn, not hardcoded in the script.
    _model = SentenceTransformer(MODEL_NAME, device="cpu")

    _qa_pairs = _load_qa_database()
    _qa_rows = _expand_with_alt_questions(_qa_pairs)
    _qa_embeddings = _build_or_load_embeddings(_qa_rows)

    _exact_match_index = {}
    for text, pair in _qa_rows:
        _exact_match_index[_normalize(text)] = pair

    print(f"Loaded {len(_qa_pairs)} Q&A pairs ({len(_qa_rows)} phrasings incl. alt_questions) and embeddings ({_qa_embeddings.shape}).")
    yield


app = FastAPI(title="MyFinergy Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _matches_stage(item_stage, target_stage: str) -> bool:
    if not item_stage or not target_stage:
        return False
    item_s = str(item_stage).upper().strip()
    target_s = str(target_stage).upper().strip()
    valid_matches = STAGE_MAPPING.get(target_s, [target_s])
    return item_s in valid_matches  # exact match against known aliases, not loose substring


def _log_unanswered(question: str, best_match: str, score: float):
    file_exists = os.path.exists(UNANSWERED_LOG_PATH)
    with open(UNANSWERED_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Question", "ClosestMatch", "Score"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question, best_match, f"{score:.3f}",
        ])


def find_answer(user_question: str) -> dict:
    normalized = _normalize(user_question)

    if normalized in START_PHRASES:
        return {"answer": START_ANSWER, "matched_question": "Getting Started / Overview",
                "stage": None, "score": 1.0, "match_type": "intent_start"}

    exact = _exact_match_index.get(normalized)
    if exact:
        return {"answer": exact["answer"], "matched_question": exact["question"],
                "stage": exact.get("stage"), "score": 1.0, "match_type": "exact"}

    query_embedding = _generate_embeddings([user_question])[0]
    scores = _qa_embeddings @ query_embedding
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_phrasing, best_pair = _qa_rows[best_idx]

    if best_score >= SIMILARITY_THRESHOLD:
        return {
            "answer": best_pair["answer"],
            "matched_question": best_pair["question"],
            "stage": best_pair.get("stage"),
            "score": best_score,
            "match_type": "semantic" if best_score >= LOW_CONFIDENCE_THRESHOLD else "semantic_low_confidence",
        }

    _log_unanswered(user_question, best_phrasing, best_score)
    return {"answer": FALLBACK_ANSWER, "matched_question": None,
            "stage": None, "score": best_score, "match_type": "none"}


@app.get("/")
def read_root():
    return {"message": "MyFinergy Chatbot API is running", "qa_pairs_loaded": len(_qa_pairs)}


@app.get("/api/faqs")
def get_faqs_by_stage(stage: Optional[str] = Query(None)):
    """
    No 'stage' -> all Q&A pairs. 'stage' (T1/T2/T3/T4) -> only that stage's
    real, tagged questions. No arbitrary fallback slicing -- if a stage
    genuinely has no tagged questions, it returns an empty list rather than
    guessing with unrelated content.
    """
    if stage:
        return [p for p in _qa_pairs if _matches_stage(p.get("stage"), stage)]
    return _qa_pairs


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_query = request.question.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = find_answer(user_query)
    return {
        "answer": result["answer"],
        "matched_question": result["matched_question"],
        "stage": result["stage"],
        "confidence": round(result["score"], 3),
        "match_type": result["match_type"],
    }


@app.post("/feedback")
def receive_feedback(data: FeedbackRequest):
    try:
        file_exists = os.path.exists(FEEDBACK_LOG_PATH)
        with open(FEEDBACK_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Question", "Answer", "Feedback"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.question, data.answer, data.feedback,
            ])
        return {"status": "success", "message": "Feedback recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@app.get("/qa-count")
def qa_count():
    return {"count": len(_qa_pairs)}