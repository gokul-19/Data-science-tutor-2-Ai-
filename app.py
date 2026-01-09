import random
import requests
import streamlit as st
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --------------------------------------------------
# Streamlit Page Config (FIRST Streamlit command)
# --------------------------------------------------
st.set_page_config(
    page_title="DataSage - AI Data Science Tutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# API Keys (Streamlit Secrets)
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
    st.error(f"Gemini initialization failed: {e}")
    st.stop()

# --------------------------------------------------
# Unsplash Image Fetcher
# --------------------------------------------------
def get_unsplash_image():
    queries = [
        "3d robot assistant",
        "3d ai chatbot",
        "futuristic ai hologram",
        "3d digital assistant"
    ]
    query = random.choice(queries)
    url = f"https://api.unsplash.com/photos/random?query={query}&client_id={UNSPLASH_API_KEY}"

    try:
        data = requests.get(url, timeout=10).json()
        return data.get("urls", {}).get("regular", "")
    except Exception:
        return ""

# --------------------------------------------------
# CSS
# --------------------------------------------------
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
.user {
    background: rgba(0,123,255,0.25);
    padding: 15px;
    border-radius: 20px 20px 0 20px;
    margin: 10px 0;
    max-width: 80%;
}
.ai {
    background: rgba(255,255,255,0.15);
    padding: 15px;
    border-radius: 20px 20px 20px 0;
    margin: 10px 0;
    max-width: 80%;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Console")

    profile_label = st.selectbox(
        "Your Expertise Level",
        ["👶 Beginner", "👨‍💻 Intermediate", "🧙‍♂️ Advanced"],
        index=1
    )

# --------------------------------------------------
# Layout
# --------------------------------------------------
img = get_unsplash_image()
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h1 style='text-align:center;'>DataSage 🧠</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>AI Data Science Tutor</p>", unsafe_allow_html=True)

with col2:
    if img:
        st.image(img, use_container_width=True)

# --------------------------------------------------
# Prompt + Chain (LATEST LangChain)
# --------------------------------------------------
prompt = PromptTemplate(
    input_variables=["question", "level"],
    template="""
You are a friendly and expert Data Science tutor.

ONLY answer questions related to:
Data Science, Machine Learning, AI, Python, Statistics, SQL, Visualization.

User level: {level}

Question:
{question}

Give a clear, structured explanation with examples.
"""
)

chain = prompt | llm

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

level = profile_label.split(" ", 1)[1]

# --------------------------------------------------
# Input
# --------------------------------------------------
question = st.text_input(
    "",
    placeholder="Ask anything about Data Science, ML, Python, Statistics..."
)

send = st.button("Send 🚀")

if question and send:
    try:
        res = chain.invoke({"question": question, "level": level})
        answer = res.content if hasattr(res, "content") else str(res)

        st.session_state.chat.append((question, answer))
    except Exception as e:
        st.error(e)

# --------------------------------------------------
# Chat Display
# --------------------------------------------------
st.markdown("### 💬 Conversation")

if st.session_state.chat:
    for q, a in st.session_state.chat:
        st.markdown(f"<div class='user'><b>You:</b> {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai'><b>DataSage:</b> {a}</div>", unsafe_allow_html=True)
else:
    st.info("👋 Ask your first Data Science question!")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<hr>
<p style="text-align:center;font-size:12px;">
🧠 Gemini • 🖼️ Unsplash • 🚀 Streamlit
</p>
""", unsafe_allow_html=True)
