import random
import requests
import streamlit as st
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="DataSage - AI Data Science Tutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# API Keys
# -----------------------------
GEMINI_API_KEY = "AIzaSyAnA1OkxkfzqnROPHb0nI0m09R7uxlHmWY"  # Your key
UNSPLASH_API_KEY = st.secrets["UNSPLASH_API_KEY"]

# -----------------------------
# Initialize Gemini
# -----------------------------
try:
    genai.configure(api_key=GEMINI_API_KEY)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Stable model
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )
except Exception as e:
    st.error(f"❌ Gemini initialization failed: {e}")
    st.stop()

# -----------------------------
# Unsplash Image Fetcher
# -----------------------------
def get_unsplash_image():
    queries = [
        "3d ai assistant",
        "futuristic ai robot",
        "3d hologram ai",
        "3d digital assistant"
    ]
    q = random.choice(queries)
    url = f"https://api.unsplash.com/photos/random?query={q}&client_id={UNSPLASH_API_KEY}"
    try:
        data = requests.get(url, timeout=10).json()
        return data.get("urls", {}).get("regular", "")
    except Exception:
        return ""

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.user-s {background:rgba(0,123,255,0.25);padding:15px;border-radius:20px;margin:8px 0;max-width:80%}
.ai-s {background:rgba(255,255,255,0.15);padding:15px;border-radius:20px;margin:8px 0;max-width:80%}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Profile
# -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Profile")
    profile_label = st.selectbox("Expertise Level", ["Beginner", "Intermediate", "Advanced"], index=1)

# -----------------------------
# Layout
# -----------------------------
img_url = get_unsplash_image()
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("<h1 style='text-align:center;'>DataSage 🧠</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Your AI Data Science Tutor</p>", unsafe_allow_html=True)
with col2:
    if img_url:
        st.image(img_url, use_container_width=True)

# -----------------------------
# Prompt + Chain
# -----------------------------
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert AI Data Science tutor.

Only answer questions related to:
Data Science, Machine Learning, AI, Python, Statistics, SQL, Visualization.

Question:
{question}

Provide a detailed explanation with examples.
"""
)
chain = prompt | llm

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# User Input
# -----------------------------
user_question = st.text_input(
    "Ask a question about Data Science, ML, Python, etc.",
    ""
)
send_button = st.button("Send 🚀")

if user_question and send_button:
    try:
        res = chain.invoke({"question": user_question})
        answer = res.content if hasattr(res, "content") else str(res)
    except Exception as e:
        # ✅ Handle quota errors gracefully
        if "RESOURCE_EXHAUSTED" in str(e):
            answer = "⚠️ Gemini API quota exceeded. Please wait or upgrade your plan."
        else:
            answer = f"⚠️ Gemini API error: {e}"

    st.session_state.chat_history.append({"q": user_question, "a": answer})

# -----------------------------
# Display Chat History
# -----------------------------
st.markdown("## 💬 Conversation")
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(f"<div class='user-s'><b>You:</b> {chat['q']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-s'><b>DataSage:</b> {chat['a']}</div>", unsafe_allow_html=True)
else:
    st.info("👋 Ask your first question about Data Science!")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr>
<p style="text-align:center;font-size:12px;">
🧠 Powered by Gemini 2.0 • 🖼️ Unsplash • 🚀 Streamlit
</p>
""", unsafe_allow_html=True)
