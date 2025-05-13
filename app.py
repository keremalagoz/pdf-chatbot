import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader # PDF okumak için
from langchain.text_splitter import RecursiveCharacterTextSplitter # Metin bölmek için
from langchain_openai import OpenAIEmbeddings # Embedding oluşturmak için
from langchain_community.vectorstores import FAISS # Vektör veritabanı için
from langchain.chains.question_answering import load_qa_chain # Soru cevap zinciri
from langchain_core.prompts import PromptTemplate # Prompt şablonu için

# --- Streamlit Secrets ve OpenRouter Konfigürasyonu ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
# LLM modeli (sohbet için)
LLM_MODEL_NAME = st.secrets.get("LLM_MODEL_NAME", "mistralai/mistral-7b-instruct:free")
# Embedding modeli (metinleri vektöre çevirmek için, OpenRouter'da OpenAI uyumlu bir model olmalı)
# Genellikle text-embedding-ada-002 veya benzeri bir model OpenRouter'da bulunur.
# OpenRouter model listesinden uygun bir embedding modeli seçin.
EMBEDDING_MODEL_NAME = st.secrets.get("EMBEDDING_MODEL_NAME", "openai/text-embedding-ada-002") # VEYA "text-embedding-ada-002"

if not OPENROUTER_API_KEY:
    st.error("OpenRouter API anahtarı bulunamadı! Lütfen Streamlit Secrets bölümüne 'OPENROUTER_API_KEY' olarak ekleyin.")
    st.stop()

# OpenRouter için OpenAI istemcisi (HEM LLM HEM DE EMBEDDING İÇİN KULLANILACAK)
# Langchain OpenAIEmbeddings, bu client'ı doğrudan kullanamayabilir,
# bu yüzden embedding için ayrıca parametreleri geçeceğiz.
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Langchain için OpenAI Embeddings yapılandırması (OpenRouter üzerinden)
# Not: Langchain'in OpenAIEmbeddings sınıfı bazen 'deployment' veya 'model' bekler.
# OpenRouter için model adını doğrudan 'model' parametresi ile veriyoruz.
try:
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        # OpenRouter'a özel başlıkları göndermek için (isteğe bağlı ama önerilir)
        # headers={
        # "HTTP-Referer": st.secrets.get("YOUR_SITE_URL", "http://localhost:8501"),
        # "X-Title": st.secrets.get("YOUR_APP_NAME", "Streamlit PDF Chatbot")
        # }
        # Bazı Langchain versiyonları 'chunk_size' bekleyebilir, gerekirse ekleyin.
        # chunk_size=1000 # Örneğin
    )
except Exception as e:
    st.error(f"Embedding modeli yüklenirken hata oluştu: {e}")
    st.info(f"Kullanılan Embedding Modeli: {EMBEDDING_MODEL_NAME}. Bu modelin OpenRouter'da 'openai/' prefix'i olmadan da (örn: 'text-embedding-ada-002') mevcut olup olmadığını kontrol edin.")
    st.stop()


# --- Yardımcı Fonksiyonlar ---
def get_pdf_text(pdf_docs):
    """Yüklenen PDF dosyalarından metinleri çıkarır."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or "" # Sayfada metin yoksa boş string ekle
    return text

def get_text_chunks(text):
    """Metni daha küçük, yönetilebilir parçalara böler."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Her bir chunk'ın maksimum karakter sayısı
        chunk_overlap=200,  # Chunk'lar arası örtüşme miktarı
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
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Vektör deposu oluşturulurken hata: {e}")
        st.error(f"Muhtemel Neden: Embedding modeli '{EMBEDDING_MODEL_NAME}' OpenRouter'da bulunamadı veya API anahtarınızla ilgili bir sorun var.")
        st.info("OpenRouter model listesini kontrol edin ve 'EMBEDDING_MODEL_NAME' secret'ını doğru ayarladığınızdan emin olun.")
        return None

def get_conversational_chain():
    """Soru-cevap için LLM zincirini oluşturur ve yapılandırır."""
    prompt_template = """
    Sadece aşağıda verilen bağlamdaki bilgileri kullanarak soruyu yanıtlayın.
    Eğer cevap bağlamda yoksa, "Bilmiyorum, bu bilgi belgede bulunmuyor." deyin.
    Kesinlikle bağlam dışı bilgi kullanmayın veya cevap uydurmayın.

    Bağlam:
    {context}

    Soru: {question}

    Cevap:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Langchain'in load_qa_chain'i doğrudan OpenAI client'ı almaz,
    # LLM'i Langchain formatında sarmallamamız gerekebilir veya
    # doğrudan OpenRouter API'sine istek atmak için bir custom chain yazabiliriz.
    # Şimdilik, Langchain'in temel LLM'leriyle uyumlu bir yapı kullanalım
    # ve OpenRouter'ı Langchain'in OpenAI LLM sarmalayıcısıyla kullanmaya çalışalım.
    
    # NOT: load_qa_chain normalde Langchain LLM nesnesi bekler.
    # Biz OpenRouter kullandığımız için, yanıtı kendimiz oluşturacağız.
    # Bu fonksiyon şimdilik sadece prompt'u döndürecek,
    # asıl LLM çağrısını ana kodda yapacağız.
    # Daha gelişmiş bir çözüm için Langchain CustomLLM veya doğrudan API çağrısı gerekir.
    return prompt


# --- Streamlit Arayüzü ---
st.set_page_config(page_title="PDF Destekli Chatbot", page_icon="📄")
st.header("📄 PDF Kaynaklı Chatbot")
st.write("Sadece yüklediğiniz PDF(ler) içeriğinden sorular sorun.")

# Session state'de değişkenleri başlatma
if "conversation_chain_prompt" not in st.session_state:
    st.session_state.conversation_chain_prompt = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

with st.sidebar:
    st.subheader("PDF Dosyalarınız")
    pdf_docs = st.file_uploader("PDF dosyalarınızı buraya yükleyin ve 'İşle' butonuna tıklayın", accept_multiple_files=True, type="pdf")

    if st.button("PDF'leri İşle"):
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
                            st.session_state.vector_store = get_vector_store(text_chunks)
                            if st.session_state.vector_store:
                                st.session_state.conversation_chain_prompt = get_conversational_chain()
                                st.session_state.chat_history = [] # PDF değişince sohbeti sıfırla
                                st.session_state.pdf_processed = True
                                st.success("PDF(ler) başarıyla işlendi! Artık soru sorabilirsiniz.")
                            else:
                                st.error("Vektör deposu oluşturulamadı. Lütfen hata mesajlarını kontrol edin.")
                                st.session_state.pdf_processed = False
                except Exception as e:
                    st.error(f"PDF işlenirken bir hata oluştu: {e}")
                    st.session_state.pdf_processed = False
        else:
            st.warning("Lütfen en az bir PDF dosyası yükleyin.")

    if st.session_state.pdf_processed:
        if st.button("Sohbeti Temizle ve PDF'i Unut"):
            st.session_state.vector_store = None
            st.session_state.conversation_chain_prompt = None
            st.session_state.chat_history = []
            st.session_state.pdf_processed = False
            st.rerun() # Sayfayı yeniden yükleyerek arayüzü temizle

st.sidebar.markdown("---")
st.sidebar.info(f"LLM Modeli: {LLM_MODEL_NAME}")
st.sidebar.info(f"Embedding Modeli: {EMBEDDING_MODEL_NAME}")


# Sohbet geçmişini gösterme
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan girdi alma
if prompt := st.chat_input("PDF içeriği hakkında sorun..."):
    if not st.session_state.pdf_processed or not st.session_state.vector_store:
        st.warning("Lütfen önce bir PDF yükleyip işleyin.")
    else:
        # Kullanıcının mesajını geçmişe ekle ve göster
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot'un yanıtını alma
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                # 1. Benzerlik Araması (Retrieval)
                # Kullanıcının sorusuna en çok benzeyen metin parçalarını vektör deposundan bul
                docs = st.session_state.vector_store.similarity_search(query=prompt, k=4) # En iyi 4 chunk
                
                if not docs:
                    full_response = "Belgede sorunuzla ilgili bir bilgi bulamadım."
                else:
                    # 2. Prompt'u Hazırlama (Augmentation)
                    # Bulunan metin parçalarını bağlam olarak kullanarak prompt'u oluştur
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Sistem mesajını ve kullanıcı prompt'unu birleştir
                    # OpenRouter için mesaj listesi formatı
                    final_prompt_messages = [
                        {"role": "system", "content": st.session_state.conversation_chain_prompt.template.format(context="DİKKAT: Bu bir yer tutucudur, asıl context aşağıdadır.", question="DİKKAT: Bu bir yer tutucudur, asıl soru aşağıdadır.")},
                        {"role": "user", "content": f"Bağlam:\n{context_text}\n\nSoru: {prompt}\n\nCevap:"}
                    ]
                    # st.write("LLM'e gönderilen mesajlar:", final_prompt_messages) # Debug için

                    # 3. Yanıt Üretme (Generation)
                    response_stream = llm_client.chat.completions.create(
                        model=LLM_MODEL_NAME,
                        messages=final_prompt_messages,
                        stream=True,
                        # extra_headers={ # Gerekirse OpenRouter'a özel başlıklar
                        # "HTTP-Referer": st.secrets.get("YOUR_SITE_URL", "http://localhost:8501"),
                        # "X-Title": st.secrets.get("YOUR_APP_NAME", "Streamlit PDF Chatbot")
                        # }
                    )
                    
                    for chunk in response_stream:
                        if chunk.choices[0].delta and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)

            except openai.APIError as e: # openai kütüphanesi artık bu şekilde
                st.error(f"OpenRouter API Hatası: {e}")
                st.error(f"Detay: {e.body if hasattr(e, 'body') else 'Detay yok'}")
                full_response = "Üzgünüm, API ile iletişimde bir sorun oluştu."
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Beklenmedik bir hata oluştu: {e}")
                import traceback
                st.error(traceback.format_exc()) # Tam hata izini göster
                full_response = "Üzgünüm, bir hata oluştu."
                message_placeholder.markdown(full_response)

        # Bot'un yanıtını geçmişe ekle
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
