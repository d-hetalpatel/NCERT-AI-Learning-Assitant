import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="NCERT Learning Assistant", layout="wide")
st.title("📘 NCERT Learning Assistant")
st.write("Select topics to get chapter, book, and research paper recommendations")

# ─────────────────────────────────────────────
# DATABASE PATH (Robust)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ncert.db")

# ─────────────────────────────────────────────
# CACHE DATABASE CONNECTION
# ─────────────────────────────────────────────
@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

conn = get_connection()

# ─────────────────────────────────────────────
# CACHE METADATA
# ─────────────────────────────────────────────
@st.cache_data
def load_metadata():
    query = "SELECT * FROM metadata"
    return pd.read_sql(query, conn)

metadata_df = load_metadata()

# ─────────────────────────────────────────────
# CACHE EMBEDDINGS
# ─────────────────────────────────────────────
@st.cache_data
def load_embeddings():
    query = "SELECT id, embedding FROM embeddings"
    df = pd.read_sql(query, conn)
    
    # Convert string → numpy array only ONCE
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(eval(x)) if isinstance(x, str) else x
    )
    
    return df

embeddings_df = load_embeddings()

# ─────────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────────
topic = st.text_input("Enter a topic")

if topic:
    st.write("Searching for:", topic)

    # Example similarity placeholder
    st.write("Metadata rows loaded:", len(metadata_df))
    st.write("Embeddings loaded:", len(embeddings_df))
