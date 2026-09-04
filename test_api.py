import requests

BASE_URL = "https://myfinergy-backend.onrender.com"

# 1. Test Root Endpoint
root_res = requests.get(f"{BASE_URL}/")
print("Root Response:", root_res.json())

# 2. Test Chat Endpoint
chat_payload = {"question": "What is myFinergy?"}
chat_res = requests.post(f"{BASE_URL}/chat", json=chat_payload)
print("Chat Response:", chat_res.json())