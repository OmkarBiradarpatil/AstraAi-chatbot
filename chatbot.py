

import os
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker


#  Load Environment / API Keys

load_dotenv()  



#  SQLite Database Setup

DB_URL = "sqlite:///chat_history.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    role = Column(String(16))  # "user" or "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)


def save_msg(role: str, content: str) -> None:
    session = SessionLocal()
    session.add(ChatMessage(role=role, content=content))
    session.commit()
    session.close()


def load_history(limit: int = 50):
    session = SessionLocal()
    records = (
        session.query(ChatMessage)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    session.close()
    messages = []
    for r in records:
        if r.role == "user":
            messages.append(HumanMessage(content=r.content))
        else:
            messages.append(AIMessage(content=r.content))
    return messages


def clear_db() -> None:
    session = SessionLocal()
    session.query(ChatMessage).delete()
    session.commit()
    session.close()




def inject_css():
    css = """
    <style>
    * {
        font-family: 'Comic Sans MS', sans-serif !important;
    }

    body, .stApp {
        background: radial-gradient(circle at top, #1a1a1a 0, #000000 55%);
        color: #f9fafb !important;
    }

    .astratitle {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        text-align: center;
        background: linear-gradient(90deg, #9b4dff, #d64fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 18px rgba(160, 78, 255, 0.95);
        margin-bottom: -4px;
    }

    .settings-box {
        background: rgba(15,15,15,0.9);
        padding: 18px 16px;
        border-radius: 14px;
        border: 1px solid rgba(148,163,184,0.5);
    }

    .settings-title {
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .label-small {
        font-size: 0.92rem;
        font-weight: 600;
        margin-top: 10px;
    }

    .stChatMessage[data-testid="user"] {
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
    }

    .stChatMessage[data-testid="assistant"] {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
    }

    .stChatInput textarea {
        background: #151515 !important;
        border: 1px solid #333 !important;
        border-radius: 999px !important;
        color: #f9fafb !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #9b4dff, #d64fff);
        border: none;
        color: white !important;
        padding: 0.6rem 1.1rem;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(174,47,255,0.55);
    }

    .stButton>button:hover {
        transform: scale(1.04);
        box-shadow: 0 0 16px rgba(174,47,255,0.9);
    }

    #MainMenu, footer {visibility: hidden;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)



#  Typing Effect
def typing_effect(text: str, placeholder, delay: float = 0.015) -> None:
    current = ""
    for word in text.split(" "):
        current += word + " "
        placeholder.markdown(current + "▌")
        time.sleep(delay)
    placeholder.markdown(text)



#  Page Config + Header
st.set_page_config(page_title="AstraAI", page_icon="✨", layout="wide")
inject_css()

st.markdown('<div class="astratitle">✨ AstraAI</div>', unsafe_allow_html=True)
st.caption("AstraAI — Speed. Clarity. Intelligence.")
st.markdown(
    "<p style='text-align:center;'>Crafted with ❤️ by <b>Omkar Biradarpatil</b></p>",
    unsafe_allow_html=True,
)
st.markdown("")


#  System Prompt (Persona)

def system_prompt(mode: str) -> str:
    if mode == "Teacher Agent":
        return (
            "You are an expert Computer Science instructor. "
            "Explain concepts step-by-step using simple language and analogies."
        )
    if mode == "Coder Agent":
        return (
            "You are a senior software engineer. Provide clean code, best practices, "
            "and concise explanations."
        )
    return (
        "You are a friendly and helpful AI assistant. "
        "Answer clearly and accurately with a supportive tone."
    )


MODEL_NAME = "llama-3.3-70b-versatile"



#  Layout

left_col, right_col = st.columns([1, 2.4])

# ---------- LEFT: Settings Panel ----------
with left_col:
    st.markdown('<div class="settings-box">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">⚙️ Settings</div>', unsafe_allow_html=True)

    mode = st.selectbox(
        "🧠 AI Persona",
        ["General Chat", "Teacher Agent", "Coder Agent"],
        index=0,
    )

    temperature = st.slider("🎛️ Creativity", 0.0, 1.0, 0.25)

    st.markdown('<div class="label-small">🧹 Maintenance</div>', unsafe_allow_html=True)
    if st.button("Clear Chat History"):
        clear_db()
        if "messages" in st.session_state:
            del st.session_state["messages"]
        st.success("Chat history cleared.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ---------- RIGHT: Chat Panel ----------
with right_col:

    # Initialize / update chat history (with hidden system message at index 0)
    if "messages" not in st.session_state:
        history_messages = load_history()
        st.session_state.messages = [SystemMessage(content=system_prompt(mode))] + history_messages
    else:
        # update persona prompt
        st.session_state.messages[0] = SystemMessage(content=system_prompt(mode))

    # Display history (skip system message at index 0)
    for msg in st.session_state.messages[1:]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # Initialize LLM
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=temperature,
    )

    # Chat input & response
    user_input = st.chat_input("Ask AstraAI anything...")

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append(HumanMessage(content=user_input))
        save_msg("user", user_input)

        # Assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🤔 Thinking...")

            try:
                response = llm.invoke(st.session_state.messages)
                reply = response.content

                typing_effect(reply, placeholder)

                st.session_state.messages.append(AIMessage(content=reply))
                save_msg("assistant", reply)

            except Exception as e:
                placeholder.markdown(f"❌ Error: {e}")
