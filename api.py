import os
import re
import csv
import json
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QA_DB_PATH = os.path.join(BASE_DIR, "qa_database_150_UPDATED.json")
EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "qa_embeddings.npy")
EMBEDDINGS_CACHE_META_PATH = os.path.join(BASE_DIR, "qa_embeddings_meta.json")
UNANSWERED_LOG_PATH = os.path.join(BASE_DIR, "unanswered_questions_log.csv")
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, "feedback_log.csv")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
hf_model = SentenceTransformer(MODEL_NAME)

SIMILARITY_THRESHOLD = 0.45       
LOW_CONFIDENCE_THRESHOLD = 0.60   

START_PHRASES = {
    "how to start", "how do i start", "getting started", "get started",
    "how to begin", "how do i begin", "where to start", "how to use this",
    "what can you do", "help", "start", "hi", "hello", "hey", "greetings"
}

START_ANSWER = (
    "Welcome to myFinergy! You can ask me questions about managing your finances, "
    "tracking budgets, understanding financial concepts, FinFit diagnostics, or advisor workflows. "
    "Try asking something like: 'What is myFinergy?' or 'How do I pitch FinFit to clients?'"
)

FALLBACK_ANSWER = (
    "I don't have a confident answer for that yet in the myFinergy knowledge base. "
    "Could you rephrase the question, or check with your team lead? "
    "I've logged this question so it can be added to the database."
)

app = FastAPI(title="MyFinergy Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str


# ---------------------------------------------------------------------------
# Knowledge base + embeddings
# ---------------------------------------------------------------------------
_qa_pairs: list[dict] = []
_qa_embeddings: np.ndarray | None = None
_exact_match_index: dict[str, dict] = {}
_qa_rows: list[tuple[str, dict]] = []


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for normalized lookups."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _load_qa_database() -> list[dict]:
    with open(QA_DB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "faqs" in data:
            return data["faqs"]
        return data


def _expand_with_alt_questions(qa_pairs: list[dict]) -> list[tuple[str, dict]]:
    rows = []
    for p in qa_pairs:
        rows.append((p["question"], p))
        for alt in p.get("alt_questions", []):
            rows.append((alt, p))
    return rows


def _generate_embeddings(texts: list[str]) -> np.ndarray:
    """Generates embeddings locally using SentenceTransformer and normalizes them."""
    normalized_texts = [_normalize(t) for t in texts]
    embeddings = hf_model.encode(normalized_texts, convert_to_numpy=True, show_progress_bar=False)

    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
    return embeddings.astype(np.float32)


def _build_or_load_embeddings(rows: list[tuple[str, dict]]) -> np.ndarray:
    texts = [_normalize(text) for text, _ in rows]
    fingerprint = {
        "count": len(texts),
        "texts_hash": hash(tuple(texts)),
        "model": MODEL_NAME,
    }

    if os.path.exists(EMBEDDINGS_CACHE_PATH) and os.path.exists(EMBEDDINGS_CACHE_META_PATH):
        with open(EMBEDDINGS_CACHE_META_PATH, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        if cached_meta == fingerprint:
            return np.load(EMBEDDINGS_CACHE_PATH)

    embeddings = _generate_embeddings(texts)

    np.save(EMBEDDINGS_CACHE_PATH, embeddings)
    with open(EMBEDDINGS_CACHE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f)

    return embeddings


@app.on_event("startup")
def load_resources():
    global _qa_pairs, _qa_embeddings, _exact_match_index, _qa_rows

    _qa_pairs = _load_qa_database()
    _qa_rows = _expand_with_alt_questions(_qa_pairs)
    _qa_embeddings = _build_or_load_embeddings(_qa_rows)

    _exact_match_index = {}
    for text, pair in _qa_rows:
        _exact_match_index[_normalize(text)] = pair

    total_phrasings = len(_qa_rows)
    print(f"Loaded {len(_qa_pairs)} Q&A pairs ({total_phrasings} phrasings incl. alt_questions) and embeddings ({_qa_embeddings.shape}).")


def _log_unanswered(question: str, best_match: str, score: float):
    file_exists = os.path.exists(UNANSWERED_LOG_PATH)
    with open(UNANSWERED_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Question", "ClosestMatch", "Score"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question,
            best_match,
            f"{score:.3f}",
        ])


def find_answer(user_question: str) -> dict:
    normalized = _normalize(user_question)

    if normalized in START_PHRASES:
        return {
            "answer": START_ANSWER,
            "matched_question": "Getting Started / Overview",
            "score": 1.0,
            "match_type": "intent_start",
        }

    exact = _exact_match_index.get(normalized)
    if exact:
        return {
            "answer": exact["answer"],
            "matched_question": exact["question"],
            "score": 1.0,
            "match_type": "exact",
        }

    query_embedding = _generate_embeddings([user_question])[0]
    scores = _qa_embeddings @ query_embedding
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_phrasing, best_pair = _qa_rows[best_idx]

    if best_score >= SIMILARITY_THRESHOLD:
        return {
            "answer": best_pair["answer"],
            "matched_question": best_pair["question"],
            "score": best_score,
            "match_type": "semantic" if best_score >= LOW_CONFIDENCE_THRESHOLD else "semantic_low_confidence",
        }

    _log_unanswered(user_question, best_phrasing, best_score)
    return {
        "answer": FALLBACK_ANSWER,
        "matched_question": None,
        "score": best_score,
        "match_type": "none",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "MyFinergy Chatbot API is running", "qa_pairs_loaded": len(_qa_pairs)}


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_query = request.question.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = find_answer(user_query)
    return {
        "answer": result["answer"],
        "matched_question": result["matched_question"],
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
                data.question,
                data.answer,
                data.feedback,
            ])
        return {"status": "success", "message": "Feedback recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@app.get("/qa-count")
def qa_count():
    return {"count": len(_qa_pairs)}