import os
import re
import csv
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QA_DB_PATH = os.path.join(BASE_DIR, "qa_database.json")
EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "qa_embeddings.npy")
UNANSWERED_LOG_PATH = os.path.join(BASE_DIR, "unanswered_questions_log.csv")
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, "feedback_log.csv")

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
    stage: Optional[str] = None

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str

_qa_pairs: list[dict] = []
_qa_embeddings: np.ndarray | None = None
_exact_match_index: dict[str, dict] = {}
_qa_rows: list[tuple[str, dict]] = []

def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)

def _load_qa_database() -> list[dict]:
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _qa_pairs, _qa_embeddings, _exact_match_index, _qa_rows
    _qa_pairs = _load_qa_database()
    _qa_rows = _expand_with_alt_questions(_qa_pairs)
    
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        _qa_embeddings = np.load(EMBEDDINGS_CACHE_PATH)

    _exact_match_index = {_normalize(text): pair for text, pair in _qa_rows}
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
    return item_s in STAGE_MAPPING.get(target_s, [target_s])

def _log_unanswered(question: str):
    file_exists = os.path.exists(UNANSWERED_LOG_PATH)
    with open(UNANSWERED_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Question"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question])

def find_answer(user_question: str) -> dict:
    normalized = _normalize(user_question)

    if normalized in START_PHRASES:
        return {"answer": START_ANSWER, "matched_question": "Getting Started / Overview",
                "stage": None, "score": 1.0, "match_type": "intent_start"}

    exact = _exact_match_index.get(normalized)
    if exact:
        return {"answer": exact["answer"], "matched_question": exact["question"],
                "stage": exact.get("stage"), "score": 1.0, "match_type": "exact"}

    # Word overlap fallback matching without ONNX
    words = set(normalized.split())
    best_score = 0.0
    best_pair = None

    for text, pair in _qa_rows:
        target_words = set(_normalize(text).split())
        if not target_words:
            continue
        score = len(words & target_words) / len(words | target_words)
        if score > best_score:
            best_score = score
            best_pair = pair

    if best_score >= 0.35 and best_pair:
        return {
            "answer": best_pair["answer"],
            "matched_question": best_pair["question"],
            "stage": best_pair.get("stage"),
            "score": round(best_score, 3),
            "match_type": "text_overlap",
        }

    _log_unanswered(user_question)
    return {"answer": FALLBACK_ANSWER, "matched_question": None,
            "stage": None, "score": 0.0, "match_type": "none"}

@app.get("/")
def read_root():
    return {"message": "MyFinergy Chatbot API is running", "qa_pairs_loaded": len(_qa_pairs)}

@app.get("/api/faqs")
@app.get("/faqs")
def get_faqs_by_stage(stage: Optional[str] = Query(None)):
    if stage:
        return [p for p in _qa_pairs if _matches_stage(p.get("stage"), stage)]
    return _qa_pairs

@app.post("/api/chat")
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
        "confidence": result["score"],
        "match_type": result["match_type"],
    }

@app.post("/api/feedback")
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

@app.get("/api/qa-count")
@app.get("/qa-count")
def qa_count():
    return {"count": len(_qa_pairs)}