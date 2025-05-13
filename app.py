import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Doğru import
from langchain_community.vectorstores import FAISS # FAISS için bu hala langchain_community'de
from langchain_core.prompts import PromptTemplate
import traceback # Hata ayıklama için

# HuggingFace Tokenizers için olası bir ayar (sorun devam ederse denenebilir)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Streamlit Secrets ve OpenRouter Konfigürasyonu ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
LLM_MODEL_NAME = st.secrets.get("LLM_MODEL_NAME", "mistralai/mistral-7b-instruct:free")
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
@st.cache_resource # Embedding modelini cache'le
def load_embeddings_model(model_name):
    try:
        st.write(f"Yerel embedding modeli yükleniyor: {model_name}")
        # model_kwargs = {'device': 'cpu'} # CPU'da çalışmasını zorlamak için
        embeddings_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            # model_kwargs=model_kwargs
        )
        st.write("Yerel embedding modeli başarıyla yüklendi.")
        return embeddings_instance
    except Exception as e:
        st.error(f"Yerel embedding modeli ({model_name}) yüklenirken hata oluştu: {e}")
        st.error(traceback.format_exc())
        st.info("Model adının doğru olduğundan ve 'sentence-transformers', 'torch' kütüphanelerinin kurulu olduğundan emin olun.")
        st.info("Popüler modeller: 'sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/all-mpnet-base-v2'")
        return None # Hata durumunda None döndür

embeddings = load_embeddings_model(LOCAL_EMBEDDING_MODEL_NAME)

if embeddings is None: # Eğer embedding modeli yüklenemezse uygulamayı durdur
    st.stop()

# --- Yardımcı Fonksiyonlar ---
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

@st.cache_resource # Vektör deposunu da cache'leyebiliriz
def create_vector_store(_text_chunks, _embeddings_model):
    if not _text_chunks:
        st.warning("PDF'ten metin çıkarılamadı veya metin boş.")
        return None
    try:
        vector_store_instance = FAISS.from_texts(texts=_text_chunks, embedding=_embeddings_model)
        return vector_store_instance
    except Exception as e:
        st.error(f"Vektör deposu oluşturulurken hata: {e}")
        st.error(traceback.format_exc())
        st.info("Embedding işlemi sırasında bir sorun oluşmuş olabilir.")
        return None

# get_conversational_chain fonksiyonu aynı kalabilir
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
st.set_page_config(page_title="PDF Destekli Chatbot", page_icon="📄")
st.header("📄 PDF Kaynaklı Chatbot")
st.write("Sadece yüklediğiniz PDF(ler) içeriğinden sorular sorun.")

if "conversation_chain_prompt" not in st.session_state:
    st.session_state.conversation_chain_prompt = get_conversational_chain_prompt()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

with st.sidebar:
    st.subheader("PDF Dosyalarınız")
    pdf_docs = st.file_uploader("PDF dosyalarınızı buraya yükleyin ve 'İşle' butonuna tıklayın", accept_multiple_files=True, type="pdf")

    if st.button("PDF'leri İşle", key="process_pdf_button"):
        if pdf_docs:
            with st.spinner("PDF'ler işleniyor... Bu işlem biraz zaman alabilir."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("PDF'lerden metin çıkarılamadı. Dosyalar boş veya okunaksız olabilir.")
                        st.session_state.pdf_processed = False
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("Metin parçalara ayrılamadı.")
                            st.session_state.pdf_processed = False
                        else:
                            # Embedding modeli zaten global 'embeddings' değişkeninde
                            st.session_state.vector_store = create_vector_store(text_chunks, embeddings)
                            if st.session_state.vector_store:
                                st.session_state.chat_history = []
                                st.session_state.pdf_processed = True
                                st.success("PDF(ler) başarıyla işlendi! Artık soru sorabilirsiniz.")
                            else:
                                st.error("Vektör deposu oluşturulamadı. Lütfen hata mesajlarını kontrol edin.")
                                st.session_state.pdf_processed = False
                except Exception as e:
                    st.error(f"PDF işlenirken bir hata oluştu: {e}")
                    st.error(traceback.format_exc())
                    st.session_state.pdf_processed = False
        else:
            st.warning("Lütfen en az bir PDF dosyası yükleyin.")

    if st.session_state.pdf_processed:
        if st.button("Sohbeti Temizle ve PDF'i Unut", key="clear_chat_button"):
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.pdf_processed = False
            # st.cache_resource.clear() # Cache'i temizlemek için, ancak dikkatli kullanılmalı
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
    if not st.session_state.pdf_processed or not st.session_state.vector_store:
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
                    
                    # Prompt'u session state'den al
                    current_prompt_template = st.session_state.conversation_chain_prompt
                    
                    # Geliştirilmiş mesaj listesi (sadece sistem ve kullanıcı mesajı)
                    # Bazı modeller context'i doğrudan sistem mesajında veya kullanıcı mesajında daha iyi anlar.
                    # Aşağıdaki format daha yaygındır.
                    messages_for_llm = [
                        {"role": "system", "content": current_prompt_template.template.split("Soru:")[0].strip()}, # Prompt'un bağlam ve genel talimat kısmı
                        {"role": "user", "content": f"Bağlam:\n{context_text}\n\nSoru: {user_query}\n\nCevap:"}
                    ]
                    
                    # st.write("LLM'e gönderilen mesajlar:", messages_for_llm) # Debug için

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
