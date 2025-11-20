import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def ask_question(query, vectordb):
    docs = vectordb.similarity_search(query, k=3)
    if not docs:
        return "Tidak ada konteks yang ditemukan di vectorstore.", []

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Gunakan konteks berikut untuk menjawab pertanyaan pengguna dengan jelas dan ringkas.

KONTEN:
{context}

PERTANYAAN:
{query}

Jawab sejelas mungkin berdasarkan konteks di atas.
"""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY belum diset di .env!")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    print("MODEL DIPAKAI :", model)

    try:
        llm = ChatGroq(model=model, api_key=api_key)
    except Exception as e:
        _handle_model_error(e, model, prefix="Error saat membuat client LLM")

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        _handle_model_error(e, model, prefix="Error saat memanggil model")

    content = getattr(response, "content", None)
    if content is None:
        try:
            content = str(response)
        except Exception:
            content = "Tidak ada respons yang dapat dibaca dari model."

    sources = [d.page_content for d in docs]
    return content, sources


def _handle_model_error(e, model, prefix="LLM Error"):
    msg = str(e)
    if any(keyword in msg for keyword in ["decommission", "model_not_found", "no longer supported"]):
        raise RuntimeError(
            f"{prefix}:\n"
            f"Model '{model}' tidak tersedia atau tidak punya akses.\n"
            f"Silakan ganti GROQ_MODEL di .env ke model valid (cek console.groq.com/docs/deprecations).\n"
            f"Detail error: {msg}"
        )
    raise RuntimeError(f"{prefix}: {msg}")
