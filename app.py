# ...existing code...
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from rag.chat import ask_question
from rag.loader import load_file
from rag.splitter import split_text
from rag.vectorstore import get_vectorstore
from rag.embedder import embedder

st.set_page_config(page_title="Talk to Your Documents", layout="wide")
st.title("📄 Talk to Your Documents – RAG Assistant")

# Pastikan GROQ_API_KEY tersedia — jika tidak, beri input di UI
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    entered = st.text_input("Masukkan GROQ API key (atau set env var GROQ_API_KEY)", type="password")
    if entered:
        os.environ["GROQ_API_KEY"] = entered
        st.success("GROQ_API_KEY diset untuk sesi ini.")

# Inisialisasi embeddings + vectorstore
embeddings = embedder()
vectordb = get_vectorstore(embeddings)

uploaded_file = st.file_uploader("Upload PDF atau TXT", type=["pdf", "txt"])

if uploaded_file:
    st.info("📥 Memproses dokumen...")
    raw_text = load_file(uploaded_file)
    chunks = split_text(raw_text)

    # Tambah chunk ke vectorstore
    try:
        vectordb.add_texts(chunks)
    except Exception as e:
        st.error(f"Gagal menambahkan teks ke vectorstore: {e}")
    else:
        st.success("Dokumen siap digunakan! Silakan bertanya.")

        query = st.text_input("Tanyakan sesuatu berdasarkan dokumen:")

        if query:
            if not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY belum diset. Masukkan API key di atas atau set environment variable GROQ_API_KEY.")
            else:
                with st.spinner("Mencari jawaban..."):
                    try:
                        answer, sources = ask_question(query, vectordb)
                        st.write("### Jawaban:")
                        st.write(answer)
                        st.write("### Sumber:")
                        st.json(sources)
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memanggil LLM: {e}")