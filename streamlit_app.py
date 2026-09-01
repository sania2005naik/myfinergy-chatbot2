# streamlit_app.py
import streamlit as st
import requests
import os
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="myFinergy - Advisor Portal",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS to match myfinergy.com styling
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar Theme - Dark Navy Slate (#0F172A) */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* Ensure ALL text, captions, and links in Sidebar are pure visible white */
    [data-testid="stSidebar"] *, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] caption {
        color: #FFFFFF !important;
    }

    /* Reset Chat History Button Styling in Sidebar */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #334155 !important;
        border-color: #0066FF !important;
        color: #38BDF8 !important;
    }

    /* Top Navigation Banner - Matching Sidebar Color #0F172A */
    .finergy-banner {
        background-color: #0F172A !important;
        padding: 1.2rem 2rem;
        border-radius: 12px;
        color: #FFFFFF !important;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
    }
    .finergy-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0;
    }
    .finergy-highlight {
        color: #0066FF !important;
    }
    .finergy-badge {
        background: rgba(0, 102, 255, 0.2);
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Clean Light Cards for Chat Bubbles */
    [data-testid="stChatMessage"] {
        padding: 1.2rem 1.4rem !important;
        border-radius: 12px !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* User Chat Bubble */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
    }
    
    /* Assistant Chat Bubble */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
    }

    /* Ensure Text in Chat Bubbles is Dark Slate and Readable */
    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] div, 
    [data-testid="stChatMessage"] span {
        color: #0F172A !important;
        font-size: 0.95rem;
    }

    /* Starter Action Pills */
    .stMain .stButton > button {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 1rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
        transition: all 0.2s ease !important;
    }
    .stMain .stButton > button:hover {
        border-color: #0066FF !important;
        color: #0066FF !important;
        background-color: #F0F6FF !important;
    }

    /* Bottom Chat Field Styling with Visible Text */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #0F172A !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748B !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Logo Check
has_logo = False
if os.path.exists("logo.png"):
    try:
        with Image.open("logo.png") as img:
            img.verify()
        has_logo = True
    except Exception:
        has_logo = False

# 4. Top Header Navbar (Same Dark Slate Navy `#0F172A` as Sidebar)
st.markdown("""
<div class="finergy-banner">
    <div class="finergy-title">my<span class="finergy-highlight">Finergy</span> AI Copilot</div>
    <div class="finergy-badge">Goal-Based Sales Assistant</div>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar Layout
with st.sidebar:
    if has_logo:
        st.markdown('<div style="background-color: #0F172A; padding: 10px; border-radius: 10px; text-align: center;">', unsafe_allow_html=True)
        st.image("logo.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("## my**Finergy**")
        
    st.caption("Sales Technology & Goal Planning Portal")
    st.divider()
    
    st.markdown("**Quick Navigation**")
    st.markdown("• Onboarding & Account Setup")
    st.markdown("• FinFit Diagnostics")
    st.markdown("• 3T Advisory Framework")
    st.markdown("• Objection Scripts")
    
    st.divider()
    
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Engine: FastAPI + ChromaDB | Status: 🟢 Online")

# 6. Initialize Chat State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Welcome to **myFinergy**! I am your AI sales and diagnostics copilot. How can I assist with your advisor workflow today?"
        }
    ]

# 7. Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 8. Starter Quick Prompt Buttons
st.markdown("**Frequently Asked Questions:**")
c1, c2, c3, c4 = st.columns(4)

selected_prompt = None

with c1:
    if st.button("What is myFinergy?", use_container_width=True):
        selected_prompt = "What is myFinergy?"
with c2:
    if st.button("FinFit Diagnostics", use_container_width=True):
        selected_prompt = "What parameters are evaluated in the FinFit Report?"
with c3:
    if st.button("3T Framework", use_container_width=True):
        selected_prompt = "What is the 3T Framework in myFinergy?"
with c4:
    if st.button("Client Objections", use_container_width=True):
        selected_prompt = "How do I handle the objection 'I need to think about it'?"

# 9. Chat Input Bar with Custom Placeholder Text
if user_input := st.chat_input("Type your question here... (e.g., How do I handle client objections?)"):
    selected_prompt = user_input

# 10. Handle API Backend Call
if selected_prompt:
    st.session_state.messages.append({"role": "user", "content": selected_prompt})
    with st.chat_message("user"):
        st.markdown(selected_prompt)

    try:
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"question": selected_prompt},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            bot_reply = data.get("answer", "No response received.")
        else:
            bot_reply = "⚠️ Connection error with API server."
    except Exception:
        bot_reply = "⚠️ API server offline. Please ensure FastAPI (`uvicorn api:app`) is running."

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.rerun()