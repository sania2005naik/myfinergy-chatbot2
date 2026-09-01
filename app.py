import streamlit as st
import requests
import os
import html

st.set_page_config(page_title="myFinergy AI Copilot", layout="wide")

st.markdown("""
<style>
    /* Dark Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b1329;
        padding-top: 1.5rem;
    }
    
    /* Sidebar Buttons */
    .stButton > button {
        background-color: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
        border-radius: 8px;
        width: 100%;
        margin-bottom: 8px;
        font-size: 15px;
    }
    .stButton > button:hover {
        border-color: #ef4444;
        color: #ef4444;
    }

    /* Main Container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 6rem;
        max-width: 1000px;
    }

    .sidebar-subtitle {
        color: #3b82f6;
        font-size: 16px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .faq-title {
        font-size: 14px;
        font-weight: 600;
        color: #94a3b8;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* Chat Bubbles */
    .user-bubble {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .bot-bubble {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: flex-start;
        gap: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .chat-text {
        color: #1e293b;
        font-size: 17px;
        line-height: 1.6;
        font-weight: 400;
    }

    .icon-user {
        background-color: #ef4444;
        color: white;
        border-radius: 8px;
        padding: 8px;
        font-size: 14px;
    }
    .icon-bot {
        background-color: #f97316;
        color: white;
        border-radius: 8px;
        padding: 8px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_GREETING = "Hello! I am your myFinergy AI Assistant. How can I help you today with onboarding, FinFit reports, or advisor workflows?"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": DEFAULT_GREETING}]

prompt_to_process = None

FAQ_INSTANT_ANSWERS = {
    "hi": "Hello! I am your myFinergy AI Assistant. How can I help you today with onboarding, FinFit reports, or advisor workflows?",
    "hello": "Hello! How can I assist you with your myFinergy advisor tools today?",
    "hey": "Hey there! Ask me anything about FinFit diagnostics, client pitching, or advisor workflows.",
    "What is myFinergy?": 
        "myFinergy is a sales-technology platform designed for financial advisors to help them move from product-based selling to goal-based financial planning.",
    "How does an advisor pitch FinFit?": 
        "Don't begin by saying 'I want to sell you insurance/investments.' Position the conversation around checking their financial fitness and goal readiness.",
    "What is FinFit and Finergy Score?": 
        "FinFit represents the client's financial fitness based on captured goals. The Finergy Score is a simplified indicator communicating overall financial strength.",
    "What is the T1, T2, T3, T4 framework?": 
        "T1 (Teach - create awareness), T2 (Test - conduct financial assessment), T3 (Treat - recommend action plan), and T4 (Track - ongoing engagement)."
}

# Sidebar
with st.sidebar:
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    else:
        st.markdown("<h2 style='color: white;'>myFinergy</h2>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-subtitle">myFinergy AI Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="faq-title">Frequently Asked Questions:</div>', unsafe_allow_html=True)

    if st.button("What is myFinergy?"):
        prompt_to_process = "What is myFinergy?"

    if st.button("Pitching FinFit to Clients"):
        prompt_to_process = "How does an advisor pitch FinFit?"

    if st.button("FinFit Diagnostic & Score"):
        prompt_to_process = "What is FinFit and Finergy Score?"

    if st.button("T1-T4 Framework Overview"):
        prompt_to_process = "What is the T1, T2, T3, T4 framework?"

    st.write("---")

    if st.button("🗑️ Reset Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": DEFAULT_GREETING}]
        st.rerun()

# User Input Box
user_input = st.chat_input("Ask anything about myFinergy platform, FinFit, or sales workflows...")
if user_input:
    prompt_to_process = user_input

# Render Chat History
for msg in st.session_state.messages:
    safe_content = html.escape(str(msg['content']))
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-bubble">
            <div class="icon-user">🤖</div>
            <div class="chat-text">{safe_content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-bubble">
            <div class="icon-bot">🏠</div>
            <div class="chat-text">{safe_content}</div>
        </div>
        """, unsafe_allow_html=True)

# Process Active Query
if prompt_to_process:
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    safe_prompt = html.escape(str(prompt_to_process))
    
    st.markdown(f"""
    <div class="user-bubble">
        <div class="icon-user">🤖</div>
        <div class="chat-text">{safe_prompt}</div>
    </div>
    """, unsafe_allow_html=True)

    clean_prompt = prompt_to_process.strip().lower().strip("!.,?")
    
    if clean_prompt in FAQ_INSTANT_ANSWERS:
        answer = FAQ_INSTANT_ANSWERS[clean_prompt]
    elif prompt_to_process in FAQ_INSTANT_ANSWERS:
        answer = FAQ_INSTANT_ANSWERS[prompt_to_process]
    else:
        with st.spinner("Generating response..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/chat", 
                    json={"question": prompt_to_process},
                    timeout=15
                )
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No response received.")
                    if isinstance(answer, list) and len(answer) > 0 and isinstance(answer[0], dict):
                        answer = answer[0].get("text", str(answer))
                else:
                    answer = f"⚠️ Backend error ({res.status_code}): {res.text}"
            except requests.exceptions.Timeout:
                answer = "⚠️ Request timed out. Backend took too long to respond."
            except Exception as e:
                answer = f"⚠️ Connection Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": str(answer)})
    st.rerun()