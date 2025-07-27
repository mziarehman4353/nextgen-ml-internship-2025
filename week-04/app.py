# app.py

import streamlit as st
import pandas as pd
import joblib
import string
import time
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Career Guidance Bot",
    page_icon="logo.png",  # favicon in browser tab
    layout="centered"
)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>Career Guidance Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ask your career questions below</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Load Model & Data ---
model = joblib.load('intent_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
df = pd.read_csv('career.csv')

# --- Sidebar Info ---
with st.sidebar:
    st.title("Developer Info")
    st.markdown("""
**Zia Ul Rehman Zafar**  
ML Intern @ NextGen  
[codewithzia.vercel.app](https://codewithzia.vercel.app)  
Built with Python, ML, Streamlit
    """)
    st.markdown("---")
    st.info("Try asking:\n- What does a data analyst do?\n- How to become a civil engineer?\n- Skills for UI/UX designer")

# --- Session State for Chat Memory ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Text Cleaner ---
def clean(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

# --- User Input ---
user_input = st.text_input("You:", placeholder="E.g. What skills do I need for digital marketing?")

if user_input:
    cleaned = clean(user_input)

    with st.spinner("Thinking..."):
        vect_input = vectorizer.transform([cleaned])
        pred_proba = model.predict_proba(vect_input)
        pred_role = model.predict(vect_input)[0]
        confidence = max(pred_proba[0]) * 100
        time.sleep(1.2)

    # Get Answer or Fallback
    if confidence > 40:
        answer = df[df['role'] == pred_role]['answer'].sample(n=1).values[0]
    else:
        answer = ("Hmm... I'm not sure. Try asking in a more detailed way like:\n\n"
                  "- *How to become a UI/UX designer?*\n"
                  "- *What does a data analyst do?*")

    # Store in Chat History
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": answer,
        "role": pred_role,
        "confidence": confidence,
        "time": timestamp
    })

# --- Display Chat History ---
st.markdown("### Chat History")

for chat in st.session_state.chat_history[::-1]:  # reverse order
    # User Message
    st.markdown(f"""
    <div style="margin-bottom: 5px;">
        <div style="color:#333;font-weight: bold;">You 
            <span style='float:right; color:#aaa;'>{chat['time']}</span>
        </div>
        <div style="background-color:#e8f0fe;padding:10px;border-radius:10px;">
            {chat['user']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bot Message
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div style="margin-top:8px;color:#333;font-weight: bold;">Bot 
            <span style='float:right; color:#aaa;'>{chat['confidence']:.1f}%</span>
        </div>
        <div style="background-color:#f9f9f9;padding:10px;border-left:4px solid #4CAF50;border-radius:10px;">
            {chat['bot']}
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 13px; color: #888;'>Built with ❤️ by Zia Ul Rehman for nextGen ML Internship Week 4</p>", unsafe_allow_html=True)
