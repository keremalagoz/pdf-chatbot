import streamlit as st
from openai import OpenAI # OpenRouter LLM için
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # <<< YENİ IMPORT
from langchain_community.vectorstores import FAISS # FAISS için bu hala langchain_community'de olabilir, kontrol edin!
                                                 # Langchain'in son versiyonlarında FAISS'in de yeri değişmiş olabilir.
                                                 # Eğer FAISS için de hata alırsanız, onun da yeni import yolunu bulmanız gerekir.
                                                 # Genellikle 'langchain_community.vectorstores.faiss' veya benzeri olur.
from langchain_core.prompts import PromptTemplate

# ... (kodun geri kalanı aynı) ...

# Embedding nesnesini oluşturduğunuz satırda hata mesajında belirtilen:
# /mount/src/chatbot/app.py:34: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings`
#  embeddings = HuggingFaceEmbeddings( # Bu satır artık yeni importu kullanmalı
#      model_name=LOCAL_EMBEDDING_MODEL_NAME,
#  )

# --- Streamlit Secrets ve OpenRouter Konfigürasyonu ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
LLM_MODEL_NAME = st.secrets.get("LLM_MODEL_NAME", "mistralai/mistral-7b-instruct:free")

# EMBEDDING_MODEL_NAME secret'ı artık yerel model adı için kullanılabilir veya sabitlenebilir.
# popüler ve hafif bir model: all-MiniLM-L6-v2
LOCAL_EMBEDDING_MODEL_NAME = st.secrets.get("LOCAL_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

if not OPENROUTER_API_KEY:
    st.error("OpenRouter API anahtarı (LLM için) bulunamadı! Lütfen Streamlit Secrets bölümüne 'OPENROUTER_API_KEY' olarak ekleyin.")
    st.stop()

# OpenRouter LLM için OpenAI istemcisi
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Langchain için HuggingFace Embeddings yapılandırması (yerel model)
try:
    st.write(f"Yerel embedding modeli yükleniyor: {LOCAL_EMBEDDING_MODEL_NAME}")
    # model_kwargs = {'device': 'cpu'} # Eğer GPU sorunu yaşarsanız veya CPU'da çalışmasını zorlamak isterseniz
    # encode_kwargs = {'normalize_embeddings': False} # Modele göre değişebilir
    embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL_NAME,
        # model_kwargs=model_kwargs,
        # encode_kwargs=encode_kwargs
    )
    st.write("Yerel embedding modeli başarıyla yüklendi.")
except Exception as e:
    st.error(f"Yerel embedding modeli ({LOCAL_EMBEDDING_MODEL_NAME}) yüklenirken hata oluştu: {e}")
    st.info("Model adının doğru olduğundan ve 'sentence-transformers' kütüphanesinin kurulu olduğundan emin olun.")
    st.info("Popüler modeller: 'sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/all-mpnet-base-v2'")
    st.stop()

# --- Yardımcı Fonksiyonlar (get_pdf_text, get_text_chunks aynı kalır) ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Metin parçalarından embedding oluşturur ve FAISS vektör deposu oluşturur."""
    if not text_chunks:
        st.warning("PDF'ten metin çıkarılamadı veya metin boş.")
        return None
    try:
        # 'embeddings' nesnesi artık HuggingFaceEmbeddings'ten geliyor
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Vektör deposu oluşturulurken hata: {e}")
        st.info("Embedding işlemi sırasında bir sorun oluşmuş olabilir.")
        return None

# get_conversational_chain fonksiyonu aynı kalabilir

# --- Streamlit Arayüzü (çoğunlukla aynı kalır) ---
# ... (önceki kodunuzdaki Streamlit arayüz kısmı) ...
# Sadece sidebar'daki embedding modeli bilgisini güncelleyebilirsiniz:
# st.sidebar.info(f"Embedding Modeli: {LOCAL_EMBEDDING_MODEL_NAME} (Yerel)")
# ... (kalan Streamlit arayüzü ve sohbet mantığı aynı) ...
