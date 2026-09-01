import os
import csv
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="MyFinergy Chatbot API")

# Enable CORS for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Data Models
class ChatRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str  # "like" or "dislike"

@app.get("/")
def read_root():
    return {"message": "MyFinergy Chatbot API is running"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_query = request.question.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    # Placeholder/Fallback AI answer processing logic
    response_text = f"This is a response for: '{user_query}'."
    return {"answer": response_text}

@app.post("/feedback")
def receive_feedback(data: FeedbackRequest):
    try:
        file_path = "feedback_log.csv"
        file_exists = os.path.exists(file_path)
        
        # Append feedback entry to CSV log
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Question", "Answer", "Feedback"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.question,
                data.answer,
                data.feedback
            ])
            
        return {"status": "success", "message": "Feedback recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")