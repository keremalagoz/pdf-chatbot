import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Google AI için yeni importlar
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS # Bu aynı kalabilir
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage # Langchain mesaj tipleri
import traceback
import uuid

# -----------------------------------------------------------------------------
# SAYFA KONFİGÜRASYONU - İLK STREAMLIT KOMUTU OLMALI!
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Google AI Destekli PDF Asistanı", page_icon="✨")
# -----------------------------------------------------------------------------

# os.environ["TOKENIZERS_PARALLELISM"] = "false" # Yerel HuggingFace için gerekliydi, Google için değil.

# --- Streamlit Secrets ve Google AI Konfigürasyonu ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
# Gemini modelleri genellikle 'gemini-pro' (metin için) veya 'gemini-1.5-flash-latest' gibi olur.
# Google AI Studio'da veya dokümantasyonda mevcut model adlarını kontrol edin.
GOOGLE_LLM_MODEL_NAME = st.secrets.get("GOOGLE_LLM_MODEL_NAME", "gemini-1.5-flash-latest")
# Google'ın embedding modeli genellikle "embedding-001" veya benzeri bir addır.
GOOGLE_EMBEDDING_MODEL_NAME = st.secrets.get("GOOGLE_EMBEDDING_MODEL_NAME", "models/embedding-001")


if not GOOGLE_API_KEY:
    st.error("Google API anahtarı bulunamadı! Lütfen Streamlit Secrets bölümüne 'GOOGLE_API_KEY' olarak ekleyin.")
    st.stop()

# LLM ve Embedding istemcilerini Google AI için yapılandırma
try:
    llm_client = ChatGoogleGenerativeAI(
        model=GOOGLE_LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1, # İsteğe bağlı, daha deterministik yanıtlar için
        # convert_system_message_to_human=True # Bazı eski Gemini modelleri için gerekebilir
    )
    print(f"Google AI LLM istemcisi '{GOOGLE_LLM_MODEL_NAME}' modeliyle bağlandı.")

    embeddings_model_global = GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY
    )
    print(f"Google AI Embedding istemcisi '{GOOGLE_EMBEDDING_MODEL_NAME}' modeliyle bağlandı.")

except Exception as e:
    st.error(f"Google AI istemcileri oluşturulurken hata: {e}")
    st.info("Google API anahtarınızın doğru olduğundan ve model adlarının geçerli olduğundan emin olun.")
    st.info(f"Kullanılan LLM Modeli: {GOOGLE_LLM_MODEL_NAME}, Embedding Modeli: {GOOGLE_EMBEDDING_MODEL_NAME}")
    st.stop()


# --- Yardımcı Fonksiyonlar (get_pdf_text, get_text_chunks aynı kalır) ---
def get_pdf_text(pdf_docs):
    # ... (öncekiyle aynı) ...
    text = ""
    if pdf_docs:
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text();
                    if page_text: text += page_text
            except Exception as e: st.warning(f"'{pdf.name}' dosyasından metin çıkarılırken hata: {e}")
    return text

def get_text_chunks(text):
    # ... (öncekiyle aynı) ...
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

# Vektör deposu oluşturma fonksiyonu artık Google embeddings kullanacak
def create_vector_store_from_chunks(text_chunks, current_embeddings_model):
    # ... (öncekiyle aynı, sadece embedding modeli farklı) ...
    if not text_chunks or not current_embeddings_model: return None
    try: return FAISS.from_texts(texts=text_chunks, embedding=current_embeddings_model)
    except Exception as e: st.error(f"Vektör deposu oluşturulurken hata: {e}"); st.error(traceback.format_exc()); return None

# Prompt şablonu aynı kalabilir, LLM'in talimatları anlaması önemlidir.
def get_conversational_chain_prompt_template():
    prompt_template_str = """
    SENİN GÖREVİN: Sadece ve sadece aşağıda "Bağlam:" olarak verilen metindeki bilgileri kullanarak "Soru:" kısmındaki soruyu yanıtlamaktır.
    KESİNLİKLE DIŞARIDAN BİLGİ KULLANMA, YORUM YAPMA, EK AÇIKLAMA EKLEME VEYA CEVAP UYDURMA.
    Cevabın SADECE ve SADECE "Bağlam:" içindeki bilgilere dayanmalıdır.

    Eğer "Soru:" kısmındaki soruya cevap "Bağlam:" içinde bulunmuyorsa, şu cevabı ver:
    "Bu bilgi sağlanan belgede bulunmuyor."
    BU CEVABIN DIŞINDA HİÇBİR ŞEY EKLEME. Örneğin, "Bu bilgi belgede yok ama genel olarak şöyledir..." GİBİ BİR AÇIKLAMA YAPMA.

    Bağlam:
    {context}

    Soru: {question}

    Cevap:"""
    return PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])

# --- Session State Başlatma ve Oturum Yönetimi Fonksiyonları (öncekiyle aynı) ---
if "sessions" not in st.session_state: st.session_state.sessions = {}
if "current_session_id" not in st.session_state: st.session_state.current_session_id = None
if "prompt_template" not in st.session_state: st.session_state.prompt_template = get_conversational_chain_prompt_template()
def create_new_session():
    session_id = str(uuid.uuid4()); session_name = f"Sohbet {len(st.session_state.sessions) + 1}"
    st.session_state.sessions[session_id] = {
        "id": session_id, "name": session_name, "pdf_names": [],
        "vector_store": None, "chat_history": [], "pdf_processed": False
    }
    st.session_state.current_session_id = session_id; return session_id
def get_active_session_data():
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.sessions:
        return st.session_state.sessions[st.session_state.current_session_id]
    return None
def delete_session(session_id_to_delete):
    if session_id_to_delete in st.session_state.sessions:
        del st.session_state.sessions[session_id_to_delete]
        if st.session_state.current_session_id == session_id_to_delete:
            st.session_state.current_session_id = None
            if st.session_state.sessions: st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]


# --- Streamlit Arayüzü (Sidebar ve PDF işleme mantığı öncekiyle büyük ölçüde aynı) ---
st.title("✨ Google AI Destekli PDF Asistanı")
# ... (Sidebar ve PDF işleme mantığı aynı, sadece embedding_model_global'i kullanıyor) ...
with st.sidebar:
    st.header("Sohbet Oturumları")
    if st.button("➕ Yeni Sohbet Başlat", key="new_chat_button"): create_new_session(); st.rerun()
    session_options = {sid: sdata["name"] for sid, sdata in st.session_state.sessions.items()}
    if not session_options and st.session_state.current_session_id is None:
        create_new_session(); session_options = {sid: sdata["name"] for sid, sdata in st.session_state.sessions.items()}
    if session_options:
        selected_session_id = st.selectbox(
            "Aktif Sohbeti Seçin:", options=list(session_options.keys()),
            format_func=lambda sid: session_options[sid],
            index=list(session_options.keys()).index(st.session_state.current_session_id) if st.session_state.current_session_id in session_options else 0,
            key="session_selector"
        )
        if selected_session_id != st.session_state.current_session_id: st.session_state.current_session_id = selected_session_id; st.rerun()
        active_session = get_active_session_data()
        if active_session:
            st.markdown("---"); st.subheader(f"Aktif: {active_session['name']}")
            uploader_key = f"pdf_uploader_{active_session['id']}"
            uploaded_pdf_docs = st.file_uploader("Bu sohbet için PDF dosyalarını yükleyin:", accept_multiple_files=True, type="pdf", key=uploader_key)
            if st.button("Seçili PDF'leri İşle", key=f"process_btn_{active_session['id']}"):
                if uploaded_pdf_docs:
                    with st.spinner("PDF'ler işleniyor..."):
                        active_session["pdf_names"] = [f.name for f in uploaded_pdf_docs]; raw_text = get_pdf_text(uploaded_pdf_docs)
                        if not raw_text.strip(): st.error("PDF'lerden metin çıkarılamadı."); active_session["vector_store"] = None; active_session["pdf_processed"] = False
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            if not text_chunks: st.error("Metin parçalara ayrılamadı."); active_session["vector_store"] = None; active_session["pdf_processed"] = False
                            else:
                                # Google embedding modelini kullan
                                vector_store = create_vector_store_from_chunks(text_chunks, embeddings_model_global)
                                if vector_store:
                                    active_session["vector_store"] = vector_store; active_session["chat_history"] = []
                                    active_session["pdf_processed"] = True; st.success(f"PDF(ler) '{active_session['name']}' için işlendi."); st.rerun()
                                else: st.error("Vektör deposu oluşturulamadı."); active_session["pdf_processed"] = False
                else: st.warning("Lütfen işlemek için PDF dosyası yükleyin.")
            if active_session["pdf_processed"] and active_session["pdf_names"]:
                 st.markdown("**İşlenmiş PDF(ler):**");
                 for name in active_session["pdf_names"]: st.caption(f"- {name}")
            st.markdown("---")
            if st.button(f"'{active_session['name']}' Oturumunu Sil", type="secondary", key=f"delete_btn_{active_session['id']}"):
                delete_session(active_session['id']); st.success(f"'{active_session['name']}' oturumu silindi."); st.rerun()
    else: st.sidebar.info("Henüz bir sohbet oturumu yok.")


# --- Ana Sohbet Alanı (LLM çağrısı Google AI'ye göre güncellendi) ---
active_session_data = get_active_session_data()
if active_session_data:
    st.subheader(f"Sohbet: {active_session_data['name']}")
    for message in active_session_data["chat_history"]:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if user_query := st.chat_input(f"'{active_session_data['name']}' hakkında sorun..."):
        if not active_session_data.get("vector_store"):
            st.warning("Bu sohbet için PDF işlenmedi/vektör deposu yok.")
        else:
            active_session_data["chat_history"].append({"role": "user", "content": user_query})
            with st.chat_message("user"): st.markdown(user_query)
            with st.chat_message("assistant"):
                message_placeholder = st.empty(); full_response_text = ""
                try:
                    docs = active_session_data["vector_store"].similarity_search(query=user_query, k=4)
                    if not docs:
                        full_response_text = "Bu bilgi sağlanan belgede bulunmuyor."
                    else:
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        current_prompt_template = st.session_state.prompt_template
                        formatted_prompt = current_prompt_template.format(context=context_text, question=user_query)
                        
                        # Google AI için Langchain mesaj formatı
                        # Gemini modelleri genellikle [HumanMessage(content="...")] listesini veya
                        # doğrudan string prompt'u kabul eder.
                        # Eğer sistem mesajı gerekiyorsa:
                        # lc_messages = [
                        # SystemMessage(content="Sen bir soru cevaplama asistanısın ve sadece verilen bağlama göre cevap verirsin."),
                        # HumanMessage(content=formatted_prompt)
                        # ]
                        # response = llm_client.invoke(lc_messages)
                        # full_response_text = response.content

                        # Streamlit için stream kullanımı daha iyi bir UX sağlar.
                        # ChatGoogleGenerativeAI .stream() metodunu destekler.
                        # Stream çıktısı genellikle AIMessageChunk objeleridir.
                        for chunk in llm_client.stream(formatted_prompt): # Tek string prompt ile
                            if hasattr(chunk, 'content'):
                                full_response_text += chunk.content
                                message_placeholder.markdown(full_response_text + "▌")
                            # Bazen ilk chunk boş gelebilir veya farklı bir yapıda olabilir, logları kontrol edin.
                    
                    message_placeholder.markdown(full_response_text)

                except Exception as e:
                    st.error(f"Google AI'den yanıt alınırken bir hata oluştu: {e}")
                    st.error(traceback.format_exc())
                    full_response_text = "Üzgünüm, bir hata oluştu."
                    message_placeholder.markdown(full_response_text)
            active_session_data["chat_history"].append({"role": "assistant", "content": full_response_text})
else:
    st.info("Lütfen bir sohbet seçin veya yeni bir tane başlatın.")

st.sidebar.markdown("---")
st.sidebar.caption(f"LLM: {GOOGLE_LLM_MODEL_NAME} (Google AI)")
st.sidebar.caption(f"Embedding: {GOOGLE_EMBEDDING_MODEL_NAME} (Google AI)")
