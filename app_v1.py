import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import traceback

# -----------------------------------------------------------------------------
# SAYFA KONFİGÜRASYONU - İLK STREAMLIT KOMUTU OLMALI!
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Destekli Chatbot", page_icon="📄")
# -----------------------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Streamlit Secrets ve OpenRouter Konfigürasyonu ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
LLM_MODEL_NAME = st.secrets.get("LLM_MODEL_NAME", "mistralai/mistral-7b-instruct:free")
LOCAL_EMBEDDING_MODEL_NAME = st.secrets.get("LOCAL_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

if not OPENROUTER_API_KEY:
    st.error("OpenRouter API anahtarı (LLM için) bulunamadı! Lütfen Streamlit Secrets bölümüne 'OPENROUTER_API_KEY' olarak ekleyin.")
    st.stop()

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Embedding modelini yükle ve cache'le
@st.cache_resource
def load_embeddings_model(model_name):
    print(f"Yerel embedding modeli yükleniyor: {model_name}")
    try:
        embeddings_instance = HuggingFaceEmbeddings(model_name=model_name)
        print("Yerel embedding modeli başarıyla yüklendi.")
        return embeddings_instance
    except Exception as e:
        st.error(f"Yerel embedding modeli ({model_name}) yüklenirken hata oluştu: {e}")
        st.error(traceback.format_exc())
        return None

embeddings_model = load_embeddings_model(LOCAL_EMBEDDING_MODEL_NAME)

if embeddings_model is None:
    st.stop()

# --- Yardımcı Fonksiyonlar ---
def get_pdf_text(pdf_docs):
    text = ""
    if pdf_docs:
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            except Exception as e:
                st.warning(f"'{pdf.name}' dosyasından metin çıkarılırken hata: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Vektör deposu oluşturma fonksiyonu artık cache'lenmiyor,
# çünkü her yeni PDF seti için yeniden oluşturulmalı.
# Embedding modeli zaten cache'leniyor.
def create_vector_store_from_chunks(text_chunks, current_embeddings_model):
    if not text_chunks:
        st.warning("Vektör deposu oluşturmak için metin parçacığı bulunamadı.")
        return None
    if not current_embeddings_model:
        st.error("Embedding modeli yüklenemedi, vektör deposu oluşturulamıyor.")
        return None
    try:
        # st.write("Vektör deposu oluşturuluyor...") # Debug için
        vector_store_instance = FAISS.from_texts(texts=text_chunks, embedding=current_embeddings_model)
        # st.write("Vektör deposu başarıyla oluşturuldu.") # Debug için
        return vector_store_instance
    except Exception as e:
        st.error(f"Vektör deposu oluşturulurken hata: {e}")
        st.error(traceback.format_exc())
        return None

def get_conversational_chain_prompt():
    prompt_template_str = """
    Sadece aşağıda verilen bağlamdaki bilgileri kullanarak soruyu yanıtlayın.
    Eğer cevap bağlamda yoksa, "Bilmiyorum, bu bilgi belgede bulunmuyor." deyin.
    Kesinlikle bağlam dışı bilgi kullanmayın veya cevap uydurmayın.

    Bağlam:
    {context}

    Soru: {question}

    Cevap:"""
    prompt = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])
    return prompt

# --- Streamlit Arayüzü ---
st.header("📄 PDF Kaynaklı Chatbot")
st.write("Sadece yüklediğiniz PDF(ler) içeriğinden sorular sorun.")

# Session state başlatmaları
if "conversation_chain_prompt" not in st.session_state:
    st.session_state.conversation_chain_prompt = get_conversational_chain_prompt()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed_files_key" not in st.session_state:
    # İşlenen PDF'lerin bir anahtarını tutarak gerçekten yeni PDF'ler mi geldiğini anlayabiliriz.
    # Örneğin, dosya adlarının birleşimi veya toplam boyutu gibi basit bir şey olabilir.
    # Şimdilik sadece varlığını kontrol eden bir flag yeterli olabilir ama bu daha sağlam.
    st.session_state.pdf_processed_files_key = None
if "current_pdf_docs_names" not in st.session_state: # İşlenen PDF adlarını saklamak için
    st.session_state.current_pdf_docs_names = []


with st.sidebar:
    st.subheader("PDF Dosyalarınız")
    uploaded_pdf_docs = st.file_uploader(
        "PDF dosyalarınızı buraya yükleyin ve 'İşle' butonuna tıklayın",
        accept_multiple_files=True,
        type="pdf",
        key="pdf_uploader" # Uploader'a bir key vererek state'ini daha iyi yönetebiliriz
    )

    if st.button("PDF'leri İşle", key="process_pdf_button"):
        if uploaded_pdf_docs:
            with st.spinner("PDF'ler işleniyor... Bu işlem biraz zaman alabilir."):
                # Önceki vektör deposunu ve sohbeti temizle
                st.session_state.vector_store = None
                st.session_state.chat_history = []
                # st.cache_resource.clear() # Bu çok genel, şimdilik kullanmayalım
                
                raw_text = get_pdf_text(uploaded_pdf_docs)
                if not raw_text.strip():
                    st.error("PDF'lerden metin çıkarılamadı. Dosyalar boş veya okunaksız olabilir.")
                    st.session_state.pdf_processed_files_key = None
                    st.session_state.current_pdf_docs_names = []
                else:
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("Metin parçalara ayrılamadı.")
                        st.session_state.pdf_processed_files_key = None
                        st.session_state.current_pdf_docs_names = []

                    else:
                        # Her zaman global `embeddings_model`'i kullan
                        new_vector_store = create_vector_store_from_chunks(text_chunks, embeddings_model)
                        if new_vector_store:
                            st.session_state.vector_store = new_vector_store
                            st.session_state.pdf_processed_files_key = "".join(sorted([f.name for f in uploaded_pdf_docs])) # Basit bir key
                            st.session_state.current_pdf_docs_names = [f.name for f in uploaded_pdf_docs]
                            st.success(f"PDF(ler) başarıyla işlendi: {', '.join(st.session_state.current_pdf_docs_names)}")
                        else:
                            st.error("Vektör deposu oluşturulamadı.")
                            st.session_state.pdf_processed_files_key = None
                            st.session_state.current_pdf_docs_names = []
        else:
            st.warning("Lütfen en az bir PDF dosyası yükleyin.")
            # Eğer dosya yüklenmemişse ve "işle"ye basılırsa, mevcut durumu sıfırla
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.pdf_processed_files_key = None
            st.session_state.current_pdf_docs_names = []
            st.info("Mevcut PDF bilgileri temizlendi.")


    if st.session_state.vector_store is not None: # Sadece PDF işlenmişse göster
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Mevcut İşlenmiş PDF(ler):**")
        for name in st.session_state.current_pdf_docs_names:
            st.sidebar.caption(f"- {name}")
        
        if st.button("Sohbeti Temizle ve PDF Bilgilerini Unut", key="clear_all_button"):
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.pdf_processed_files_key = None
            st.session_state.current_pdf_docs_names = []
            # `st.cache_resource.clear()` tüm cache'lenmiş kaynakları temizler.
            # Sadece embedding modelini etkilememek için dikkatli kullanılmalı.
            # Eğer sadece bu uygulamada embedding modeli cache'leniyorsa sorun olmaz.
            # st.cache_resource.clear() # Gerekirse bunu etkinleştirin
            st.success("Sohbet ve PDF bilgileri temizlendi.")
            # PDF uploader'ın değerini sıfırlamak için (Streamlit'in içsel state'i nedeniyle bazen zor olabilir)
            # st.experimental_rerun() # veya st.rerun()
            st.rerun()


st.sidebar.markdown("---")
st.sidebar.info(f"LLM Modeli: {LLM_MODEL_NAME}")
st.sidebar.info(f"Embedding Modeli: {LOCAL_EMBEDDING_MODEL_NAME} (Yerel)")

# Sohbet geçmişini gösterme
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan girdi alma
if user_query := st.chat_input("PDF içeriği hakkında sorun..."):
    if st.session_state.vector_store is None:
        st.warning("Lütfen önce bir PDF yükleyip işleyin.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_text = ""
            try:
                docs = st.session_state.vector_store.similarity_search(query=user_query, k=4)
                
                if not docs:
                    full_response_text = "Belgede sorunuzla ilgili bir bilgi bulamadım."
                else:
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    current_prompt_template = st.session_state.conversation_chain_prompt
                    
                    messages_for_llm = [
                        {"role": "system", "content": current_prompt_template.template.split("Soru:")[0].strip()},
                        {"role": "user", "content": f"Bağlam:\n{context_text}\n\nSoru: {user_query}\n\nCevap:"}
                    ]
                    
                    response_stream = llm_client.chat.completions.create(
                        model=LLM_MODEL_NAME,
                        messages=messages_for_llm,
                        stream=True,
                    )
                    
                    for chunk in response_stream:
                        if chunk.choices[0].delta and chunk.choices[0].delta.content:
                            full_response_text += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response_text + "▌")
                
                message_placeholder.markdown(full_response_text)

            except OpenAI.APIError as e:
                st.error(f"OpenRouter API Hatası: {e}")
                st.error(f"Detay: {e.body if hasattr(e, 'body') else 'Detay yok'}")
                full_response_text = "Üzgünüm, API ile iletişimde bir sorun oluştu."
                message_placeholder.markdown(full_response_text)
            except Exception as e:
                st.error(f"Beklenmedik bir hata oluştu: {e}")
                st.error(traceback.format_exc())
                full_response_text = "Üzgünüm, bir hata oluştu."
                message_placeholder.markdown(full_response_text)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response_text})
