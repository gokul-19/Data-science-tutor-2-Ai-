import os
import random
import requests
import streamlit as st
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# --------------------------------------------------
# Streamlit Page Config (MUST be first Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="DataSage - AI Data Science Tutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Secrets / API Keys
# --------------------------------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
UNSPLASH_API_KEY = st.secrets["UNSPLASH_API_KEY"]

# --------------------------------------------------
# Initialize Gemini
# --------------------------------------------------
try:
    genai.configure(api_key=GEMINI_API_KEY)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )
except Exception as e:
    st.error(f"❌ Gemini initialization failed: {e}")
    st.stop()

# --------------------------------------------------
# Unsplash Image Fetcher
# --------------------------------------------------
def get_unsplash_image():
    queries = [
        "3d robot assistant",
        "3d digital assistant",
        "3d ai chatbot",
        "3d futuristic ai",
        "hologram ai assistant"
    ]

    query = random.choice(queries)
    url = f"https://api.unsplash.com/photos/random?query={query}&client_id={UNSPLASH_API_KEY}"

    try:
        response = requests.get(url, timeout=10).json()
        return response.get("urls", {}).get("regular", "")
    except Exception:
        return ""

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
def load_css():
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #051937, #004d7a, #008793, #00bf72);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .user-bubble {
        background: rgba(0,123,255,0.25);
        border-radius: 20px 20px 0 20px;
        padding: 15px;
        margin: 10px 0;
        max-width: 80%;
    }
    .ai-bubble {
        background: rgba(255,255,255,0.15);
        border-radius: 20px 20px 20px 0;
        padding: 15px;
        margin: 10px 0;
        max-width: 80%;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Console")

    profiles = {
        "Beginner": "👶",
        "Intermediate": "👨‍💻",
        "Advanced": "🧙‍♂️"
    }

    profile_label = st.selectbox(
        "Your Expertise Level",
        [f"{v} {k}" for k, v in profiles.items()],
        index=1
    )

    st.markdown("### 📊 Skill Focus")
    python_lvl = st.slider("Python", 0, 100, 70)
    stats_lvl = st.slider("Statistics", 0, 100, 60)
    ml_lvl = st.slider("Machine Learning", 0, 100, 50)

# --------------------------------------------------
# Main Layout
# --------------------------------------------------
image_url = get_unsplash_image()
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <h1 style="text-align:center;">DataSage 🧠</h1>
    <p style="text-align:center;">Your AI Data Science Tutor</p>
    """, unsafe_allow_html=True)

with col2:
    if image_url:
        st.image(image_url, use_container_width=True)

# --------------------------------------------------
# Prompt + Chain (NEW LangChain Syntax)
# --------------------------------------------------
prompt = PromptTemplate(
    input_variables=["question", "profile"],
    template="""
You are an expert and friendly AI tutor for Data Science.

Only answer Data Science, Machine Learning, Statistics,
Python, SQL, AI, or Visualization questions.

User expertise level: {profile}

Question:
{question}

Give a clear explanation with examples if helpful.
"""
)

chain = prompt | llm

# --------------------------------------------------
# Chat State
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

profile = profile_label.split(" ", 1)[1]

# --------------------------------------------------
# User Input
# --------------------------------------------------
user_question = st.text_input(
    "",
    placeholder="Ask anything about Data Science, ML, Python, Statistics..."
)

send = st.button("Send 🚀")

if user_question and send:
    try:
        response = chain.invoke({
            "question": user_question,
            "profile": profile
        })

        answer = response.content if hasattr(response, "content") else str(response)

        st.session_state.chat_history.append({
            "q": user_question,
            "a": answer
        })

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# --------------------------------------------------
# Display Chat
# --------------------------------------------------
st.markdown("### 💬 Conversation")

if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(
            f"<div class='user-bubble'><b>You:</b> {chat['q']}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='ai-bubble'><b>DataSage:</b> {chat['a']}</div>",
            unsafe_allow_html=True
        )
else:
    st.info("👋 Ask your first Data Science question!")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<hr>
<p style="text-align:center; font-size:12px;">
🧠 Powered by Gemini • 🖼️ Unsplash • 🚀 Streamlit
</p>
""", unsafe_allow_html=True)
