import streamlit as st
import requests
import pandas as pd
import os

API_BASE = "http://127.0.0.1:8000"
LOG_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
DOWNLOAD_FILE = os.path.join(LOG_FOLDER, "downloads.csv")

st.set_page_config(page_title="Oil & Gas AI Search", layout="wide")
st.title("🔍 Oil & Gas AI Search")

# Sidebar: Analytics
st.sidebar.header("📊 Analytics")
if os.path.exists(DOWNLOAD_FILE):
    df = pd.read_csv(DOWNLOAD_FILE, names=["filename", "timestamp"])
    top_files = df["filename"].value_counts().head(5)
    st.sidebar.subheader("Most Downloaded Files")
    for file, count in top_files.items():
        st.sidebar.write(f"{file} ({count} downloads)")
else:
    st.sidebar.write("No downloads yet.")

# Session state for conversation memory
if 'history' not in st.session_state:
    st.session_state['history'] = []  # stores (query, summary, results)

query = st.text_input("Enter your search query:")

if st.button("Search"):
    if query:
        with st.spinner("Searching and generating summary..."):
            response = requests.get(f"{API_BASE}/search", params={"query": query})
            if response.status_code == 200:
                data = response.json()
                st.session_state['history'].append((query, data['summary'], data['results']))
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

# Display conversation history
for idx, (q, summary, results) in enumerate(st.session_state['history']):
    st.markdown(f"### 📝 Query {idx+1}: {q}")
    st.success(summary)

    st.markdown("#### 🔍 Results:")
    for res in results:
        st.subheader(f"Rank {res['rank']} - {res['filename']}")
        st.write(res['snippet'])
        download_link = f"{API_BASE}/download?filename={res['filename']}"
        st.markdown(f"[📥 Download File]({download_link})", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"👍 Helpful ({idx}_{res['rank']})"):
                requests.post(f"{API_BASE}/feedback", params={"query": q, "filename": res['filename'], "feedback": "positive"})
                st.toast(f"Feedback recorded for {res['filename']}")
        with col2:
            if st.button(f"👎 Not Helpful ({idx}_{res['rank']})"):
                requests.post(f"{API_BASE}/feedback", params={"query": q, "filename": res['filename'], "feedback": "negative"})
                st.toast(f"Feedback recorded for {res['filename']}")
