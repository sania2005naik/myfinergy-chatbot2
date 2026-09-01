from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Enable CORS so your app/web can freely call endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html directly when opening http://10.0.2.2:8000/
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

# Your existing chat endpoint
@app.post("/api/chat")
async def chat_endpoint(data: dict):
    user_message = data.get("message", "")
    # ... your existing chat response logic ...
    return {"response": f"Echo: {user_message}"}