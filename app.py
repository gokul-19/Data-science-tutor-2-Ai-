import random
import requests
import streamlit as st
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(
    page_title="DataSage - AI Data Science Tutor",
    layout="wide"
)

# --------------------------------------------------
# Secrets
# --------------------------------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
UNSPLASH_API_KEY = st.secrets["UNSPLASH_API_KEY"]

# --------------------------------------------------
# Gemini initialization (NEW MODEL)
# --------------------------------------------------
try:
    genai.configure(api_key=GEMINI_API_KEY)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",   # ✅ NEW & WORKING
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )
except Exception as e:
    st.error(f"Gemini init failed: {e}")
    st.stop()

# --------------------------------------------------
# Unsplash image
# --------------------------------------------------
def get_unsplash_image():
    queries = [
        "3d ai assistant",
        "futuristic ai robot",
        "hologram ai",
        "3d chatbot"
    ]
    q = random.choice(queries)
    url = f"https://api.unsplash.com/photos/random?query={q}&client_id={UNSPLASH_API_KEY}"
    try:
        return requests.get(url, timeout=10).json()["urls"]["regular"]
    except Exception:
        return ""

# --------------------------------------------------
# UI
# --------------------------------------------------
st.markdown("""
<style>
.user {background:#1f77b433;padding:15px;border-radius:20px 20px 0 20px;margin:8px 0}
.ai {background:#ffffff22;padding:15px;border-radius:20px 20px 20px 0;margin:8px 0}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    level_label = st.selectbox(
        "Your Level",
        ["👶 Beginner", "👨‍💻 Intermediate", "🧙‍♂️ Advanced"],
        index=1
    )

img = get_unsplash_image()
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown("<h1>DataSage 🧠</h1>", unsafe_allow_html=True)
    st.markdown("AI Data Science Tutor")
with c2:
    if img:
        st.image(img, use_container_width=True)

# --------------------------------------------------
# Prompt (LangChain NEW)
# --------------------------------------------------
prompt = PromptTemplate(
    input_variables=["question", "level"],
    template="""
You are an expert Data Science tutor.

Only answer Data Science / ML / AI / Python / Statistics questions.

User level: {level}

Question:
{question}

Give a clear explanation with examples.
"""
)

chain = prompt | llm
level = level_label.split(" ", 1)[1]

# --------------------------------------------------
# Chat state
# --------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

q = st.text_input("Ask a Data Science question")
if st.button("Send 🚀") and q:
    res = chain.invoke({"question": q, "level": level})
    ans = res.content if hasattr(res, "content") else str(res)
    st.session_state.chat.append((q, ans))

# --------------------------------------------------
# Display
# --------------------------------------------------
st.markdown("## 💬 Conversation")
for q, a in st.session_state.chat:
    st.markdown(f"<div class='user'><b>You:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='ai'><b>DataSage:</b> {a}</div>", unsafe_allow_html=True)
