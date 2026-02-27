import os
import re
import hashlib
import urllib.parse

import numpy as np
import streamlit as st
import pandas as pd
import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_PATH = "ncert.db"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="NCERT Learning Assistant", layout="wide")
st.title("📘 NCERT Learning Assistant")
st.write("Select topics to get chapter, book, and research paper recommendations for Class 11–12 students")

# ─────────────────────────────────────────────
# MOUNT GOOGLE DRIVE
# ─────────────────────────────────────────────
try:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except ImportError:
    DB_PATH = "ncert.db"  # fallback for local runs

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
@st.cache_resource
def get_db():
    if not os.path.exists(DB_PATH):
        st.error(
            f"❌ Database not found at:\n`{DB_PATH}`\n\n"
            "Please run setup_db.ipynb first, or update DB_PATH at the top of app.py."
        )
        st.stop()
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1")  # sanity check
        return conn
    except sqlite3.DatabaseError as e:
        st.error(
            f"❌ File exists but is not a valid SQLite database:\n`{DB_PATH}`\n\n"
            f"Error: {e}\n\nPlease re-run setup_db.ipynb to recreate it."
        )
        st.stop()

# ─────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────
def get_subjects() -> list:
    cur = get_db().cursor()
    cur.execute("SELECT subject_name FROM subjects ORDER BY subject_name")
    return [row["subject_name"] for row in cur.fetchall()]


def get_levels(subject_name: str = None) -> list:
    cur = get_db().cursor()
    level_order = (
        "CASE level WHEN 'Beginner' THEN 1 "
        "WHEN 'Intermediate' THEN 2 WHEN 'Advanced' THEN 3 ELSE 4 END"
    )
    if subject_name:
        subject_id = subject_name.lower().replace(" ", "_")
        cur.execute(
            f"SELECT DISTINCT level FROM recommendations "
            f"WHERE subject_id = ? ORDER BY {level_order}",
            (subject_id,)
        )
    else:
        cur.execute(f"SELECT DISTINCT level FROM recommendations ORDER BY {level_order}")
    return [row["level"] for row in cur.fetchall()]


def get_recommendations(subject_name: str, level: str) -> list:
    subject_id = subject_name.lower().replace(" ", "_")
    cur = get_db().cursor()
    cur.execute("""
        SELECT title, author, level_type, why, link, journal, ncert_chapter_link
        FROM recommendations
        WHERE subject_id = ? AND level = ?
        ORDER BY rec_id
    """, (subject_id, level))
    return [dict(row) for row in cur.fetchall()]


def get_all_topics(subject_name: str = None) -> list:
    cur = get_db().cursor()
    if subject_name:
        subject_id = subject_name.lower().replace(" ", "_")
        cur.execute("""
            SELECT DISTINCT t.topic_text FROM topics t
            WHERE t.source_pdf IN (
                SELECT DISTINCT pdf_filename FROM chapters WHERE subject_id = ?
            )
            ORDER BY t.topic_text
        """, (subject_id,))
    else:
        cur.execute("SELECT DISTINCT topic_text FROM topics ORDER BY topic_text LIMIT 600")
    return [r["topic_text"] for r in cur.fetchall()]


def lookup_chapter(pdf_filename: str) -> dict:
    cur = get_db().cursor()
    cur.execute("""
        SELECT subject_id, class, chapter_number, chapter_name, book_name, is_special
        FROM chapters WHERE pdf_filename = ? LIMIT 1
    """, (pdf_filename,))
    row = cur.fetchone()
    if row:
        return dict(row)
    return {
        "subject_id": "—", "class": "—", "chapter_number": "—",
        "chapter_name": pdf_filename, "book_name": "—", "is_special": 0
    }

# ─────────────────────────────────────────────
# EMBEDDER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Loading embedding model…")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ─────────────────────────────────────────────
# LOAD ALL EMBEDDINGS FROM DB (once per session)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading embeddings from database…")
def load_embeddings_from_db():
    """
    Reads ALL embeddings + filenames straight from DB into numpy arrays.
    No PDF reading. No encoding. Pure DB → RAM.
    Runs once per session in ~1-2 seconds.
    """
    cur = get_db().cursor()
    cur.execute("SELECT pdf_filename, embedding FROM pdf_embeddings")
    rows = cur.fetchall()

    paths      = []
    embeddings = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        paths.append(row["pdf_filename"])
        embeddings.append(emb)

    return paths, np.array(embeddings, dtype=np.float32)

# ─────────────────────────────────────────────
# CHAPTER RECOMMENDATIONS
# ─────────────────────────────────────────────
def get_chapter_recommendations(query: str, top_n: int = 5) -> pd.DataFrame:
    if not query or not query.strip():
        return pd.DataFrame()

    paths, text_emb = load_embeddings_from_db()   # instant — already in RAM
    query_emb = embedder.encode([query.strip()], convert_to_numpy=True)
    sims      = cosine_similarity(query_emb, text_emb).flatten()
    top_idx   = sims.argsort()[-top_n:][::-1]

    rows = []
    for i in top_idx:
        info = lookup_chapter(paths[i])
        if info.get("is_special"):
            continue
        rows.append({
            "Subject":      info["subject_id"].replace("_", " ").title(),
            "Class":        f"Class {info['class']}",
            "Book":         info["book_name"],
            "Ch No.":       info["chapter_number"],
            "Chapter Name": info["chapter_name"],
            "Relevance":    round(float(sims[i]), 4),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ─────────────────────────────────────────────
# YOUTUBE URL BUILDER
# ─────────────────────────────────────────────
def build_youtube_url(subject: str, level: str, topics: list) -> str:
    topic_str = " ".join(topics[:3]) if topics else ""
    query     = f"NCERT {subject} {level} {topic_str} class 11 12".strip()
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("🎓 Select Options")

all_subjects     = get_subjects()
selected_subject = st.sidebar.selectbox("Select Subject", all_subjects)

all_levels       = get_levels(selected_subject)
selected_level   = st.sidebar.selectbox("Select Level", all_levels)

st.sidebar.markdown("---")

input_mode = st.sidebar.radio(
    "Search by",
    ["📝 Type a query", "📋 Pick from topics"],
    help="Type your own question OR pick chapter names from the list"
)

user_query      = ""
selected_topics = []

if input_mode == "📝 Type a query":
    user_query = st.sidebar.text_area(
        "Your question or keyword",
        placeholder=(
            "e.g. what is the caste system?\n"
            "or: federalism in India\n"
            "or: GDP and national income"
        ),
        height=100,
        help="Type anything — a question, keyword, or concept"
    )
    selected_topics = [w for w in user_query.split() if len(w) > 3] if user_query else []

else:
    subject_topics = get_all_topics(selected_subject)
    if not subject_topics:
        st.sidebar.warning("No topics found in database.")
        selected_topics = []
    else:
        st.sidebar.caption(f"{len(subject_topics)} topics available")
        selected_topics = st.sidebar.multiselect(
            "Select Topic(s)", subject_topics,
            help="Topics from NCERT chapters"
        )
    user_query = " ".join(selected_topics)

search_query = user_query.strip()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3 = st.tabs([
    "📄 Chapter Recommendations",
    "📚 Books & Research Papers",
    "▶️ YouTube Videos",
])

# ── Tab 1: Chapter Recommendations ────────────
with tab1:
    st.markdown(
        "Finds the most relevant NCERT chapters based on your query. "
        "Works with free text — *what is caste system*, *explain GDP*, anything."
    )
    if not search_query:
        st.info("👈 Type a question or select topics from the sidebar to search.")
    else:
        st.markdown(f"🔎 **Searching for:** `{search_query}`")
        with st.spinner("Finding relevant chapters..."):
            df = get_chapter_recommendations(search_query)
        if df.empty:
            st.warning("No matching chapters found. Try different keywords.")
        else:
            st.success(f"✅ Top {len(df)} chapters found!")
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Relevance": st.column_config.ProgressColumn(
                        "Relevance", min_value=0, max_value=1, format="%.4f"
                    )
                }
            )

# ── Tab 2: Books & Research Papers ────────────
with tab2:
    st.markdown(
        f"Showing **{selected_level}** level resources for **{selected_subject}** "
        f"— NCERT textbooks → reference books → research papers."
    )
    if st.button("📖 Get Books & Papers", type="primary", key="book_btn"):
        rows = get_recommendations(selected_subject, selected_level)
        if not rows:
            st.warning(f"No recommendations found for {selected_subject} / {selected_level}")
        else:
            for r in rows:
                type_icons = {"NCERT": "📗", "Book": "📘", "Paper": "📄"}
                icon = type_icons.get(r["level_type"], "📌")
                with st.expander(f"{icon} {r['title']} — *{r['author']}*", expanded=False):
                    st.markdown(f"**Type:** {r['level_type']}")
                    if r["journal"]:
                        st.markdown(f"**Journal:** {r['journal']}")
                    if r["ncert_chapter_link"]:
                        st.markdown(f"**NCERT Chapter:** `{r['ncert_chapter_link']}`")
                    st.markdown(f"💡 *{r['why']}*")
                    if r["link"]:
                        st.markdown(f"[🔗 Open Resource]({r['link']})")

# ── Tab 3: YouTube Videos ─────────────────────
with tab3:
    st.markdown(
        "Clicking the button below will open a **YouTube search** in a new tab "
        "with a query built from your selected subject, level, and topics."
    )
    yt_query_preview = f"NCERT {selected_subject} {selected_level} {search_query} class 11 12".strip()
    st.markdown(f"**Search query:** `{yt_query_preview}`")

    if not search_query:
        st.info("👈 Type a query or select topics to refine the YouTube search.")

    yt_url = build_youtube_url(selected_subject, selected_level, [search_query])
    st.link_button("▶️ Search on YouTube", yt_url, type="primary")

    st.markdown("---")
    st.markdown("**Or jump directly to a level:**")

    level_icons = {"Beginner": "🌱", "Intermediate": "📚", "Advanced": "🔬"}
    yt_levels   = get_levels(selected_subject)
    cols        = st.columns(len(yt_levels))
    for col, lvl in zip(cols, yt_levels):
        with col:
            icon = level_icons.get(lvl, "▶️")
            url  = build_youtube_url(selected_subject, lvl, [search_query])
            st.link_button(f"{icon} {lvl}", url, use_container_width=True)
