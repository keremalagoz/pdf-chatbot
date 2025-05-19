# ✨📚 Google AI Destekli Çoklu Sohbet PDF Asistanı / Google AI Powered Multi-Session PDF Assistant 📄

Bu Streamlit uygulaması, kullanıcıların PDF dosyaları yükleyerek bu dosyaların içeriği hakkında sorular sormasına olanak tanır. Genel sorular sorulduğunda ("Bu PDF ne hakkında?"), uygulama PDF içeriğinden bir özet veya örnek sorular türetmeye çalışır. Spesifik sorular için ise, yüklenen PDF'leri kaynak olarak kullanarak yanıtlar üretir. Uygulama, her bir PDF seti için ayrı sohbet oturumları oluşturur ve yönetir.

This Streamlit application allows users to upload PDF files and ask questions about their content. When general questions are asked (e.g., "What is this PDF about?"), the application attempts to generate a summary or sample questions from the PDF content. For specific questions, it generates answers using the uploaded PDFs as a source. The application creates and manages separate chat sessions for each set of PDFs.

---

## 🇹🇷 Türkçe Açıklama

### ✨ Öne Çıkan Özellikler

*   **PDF Yükleme:** Bir veya daha fazla PDF dosyası yüklenebilir.
*   **İçerik Tabanlı Soru Cevaplama (RAG):** Yüklenen PDF'lerin içeriğine dayalı olarak spesifik sorulara yanıt verir.
*   **Genel Soru Anlama ve Yanıtlama:** "Bu PDF ne hakkında?" gibi genel sorulara, PDF'ten özet veya örnek sorular türeterek yanıt vermeye çalışır.
    *   Bu özellik için **Langchain PydanticOutputParser** kullanılarak yapılandırılmış çıktı (JSON) hedeflenir.
*   **Çoklu Sohbet Oturumları:** Her PDF seti veya başlatılan sohbet için ayrı oturumlar oluşturulabilir, seçilebilir ve yönetilebilir.
*   **Google Generative AI Entegrasyonu:**
    *   **LLM:** Metin üretimi ve soruları yanıtlama için Google'ın Gemini modelleri (örn: `gemini-1.5-flash-latest`) kullanılır.
    *   **Embedding:** Metinleri vektörlere dönüştürmek için Google'ın embedding modelleri (örn: `models/embedding-001`) kullanılır.
*   **Sohbet Geçmişi:** Her oturum için ayrı sohbet geçmişi tutulur.
*   **Kısıtlayıcı Prompt Mühendisliği:** LLM'in sadece yüklenen PDF içeriğine odaklanmasını sağlamak ve dışarıdan bilgi kullanmasını engellemek için özel olarak tasarlanmış prompt şablonları kullanılır.

### 🚀 Kurulum ve Çalıştırma

1.  **Depoyu Klonlayın:**
    ```bash
    git clone https://github.com/SENIN-KULLANICI-ADIN/SENIN-REPO-ADIN.git
    cd SENIN-REPO-ADIN
    ```

2.  **Sanal Ortam Oluşturun (Önerilir):**
    ```bash
    python -m venv venv
    # Windows için:
    venv\Scripts\activate
    # macOS/Linux için:
    source venv/bin/activate
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Google API Anahtarını Ayarlayın (Streamlit Secrets):**
    Projenizin ana dizininde `.streamlit` adında bir klasör oluşturun ve içine `secrets.toml` adında bir dosya ekleyin.
    ```toml
    # .streamlit/secrets.toml

    GOOGLE_API_KEY = "SENIN_GOOGLE_AI_STUDIO_API_ANAHTARIN"

    # İsteğe bağlı: Kullanılacak Google AI modellerini değiştirmek için
    # GOOGLE_LLM_MODEL_NAME = "gemini-1.5-flash-latest"
    # GOOGLE_EMBEDDING_MODEL_NAME = "models/embedding-001"
    ```
    **Not:** `SENIN_GOOGLE_AI_STUDIO_API_ANAHTARIN` kısmını kendi Google AI Studio'dan aldığınız API anahtarınızla değiştirin.

5.  **Uygulamayı Çalıştırın:**
    ```bash
    streamlit run app.py
    ```
    Uygulama varsayılan olarak `http://localhost:8501` adresinde açılacaktır.

### 🛠️ Nasıl Çalışır?

1.  Kullanıcı bir sohbet oturumu başlatır veya mevcut birini seçer.
2.  Seçili oturum için bir veya daha fazla PDF dosyası yükler ve "İşle" butonuna tıklar.
3.  Uygulama, PDF'lerden metinleri çıkarır ve daha küçük parçalara (chunks) böler. Bu chunk'lar hem RAG için hem de genel sorulara yanıt için saklanır.
4.  Google'ın embedding modeli kullanılarak bu metin parçaları vektörlere (embeddings) dönüştürülür.
5.  Bu vektörler, hızlı benzerlik araması için bir FAISS vektör deposunda saklanır (aktif oturuma özel).
6.  Kullanıcı bir soru sorduğunda:
    *   **Genel Soru İse:** Kullanıcının sorusu genel bir ifade içeriyorsa (örn: "PDF ne hakkında?"), saklanan metin chunk'larının bir kısmı ve yapılandırılmış çıktı (özet veya örnek sorular) için özel bir prompt şablonu kullanılarak Google Gemini LLM'ine istek gönderilir. Yanıt, PydanticOutputParser ile parse edilerek kullanıcıya sunulur.
    *   **Spesifik Soru İse (RAG):** Sorunun vektörüne en yakın olan metin parçaları (ilgili bağlam) FAISS vektör deposundan alınır. Bu bağlam ve kullanıcının sorusu, LLM'i sadece sağlanan bilgiyi kullanmaya yönlendiren kısıtlayıcı bir RAG prompt şablonu kullanılarak formatlanır. Formatlanmış prompt, Google Gemini LLM'ine gönderilir ve yanıt kullanıcıya gösterilir.
7.  Her sohbet oturumunun kendi PDF bilgisi, vektör deposu ve sohbet geçmişi ayrı olarak tutulur.

---

## 🇬🇧🇺🇸 English Description

### ✨ Key Features

*   **PDF Upload:** Allows uploading one or more PDF files.
*   **Content-Based Q&A (RAG):** Answers specific questions based on the content of the uploaded PDFs.
*   **General Query Understanding and Response:** When asked general questions (e.g., "What is this PDF about?"), it attempts to generate a summary or sample questions from the PDF content.
    *   Utilizes **Langchain PydanticOutputParser** for structured output (JSON) for this feature.
*   **Multi-Session Chat:** Create, select, and manage separate chat sessions for each set of PDFs or initiated chats.
*   **Google Generative AI Integration:**
    *   **LLM:** Uses Google's Gemini models (e.g., `gemini-1.5-flash-latest`) for text generation and answering questions.
    *   **Embedding:** Uses Google's embedding models (e.g., `models/embedding-001`) to convert text into vectors.
*   **Chat History:** Maintains a separate chat history for each session.
*   **Constrained Prompt Engineering:** Utilizes specifically designed prompt templates to ensure the LLM focuses solely on the uploaded PDF content and prevents it from using external knowledge.

### 🚀 Setup and Running

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
    cd YOUR-REPO-NAME
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # For Windows:
    venv\Scripts\activate
    # For macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Google API Key (Streamlit Secrets):**
    In your project's root directory, create a folder named `.streamlit` and add a file named `secrets.toml` inside it.
    ```toml
    # .streamlit/secrets.toml

    GOOGLE_API_KEY = "YOUR_GOOGLE_AI_STUDIO_API_KEY"

    # Optional: To change the Google AI models used
    # GOOGLE_LLM_MODEL_NAME = "gemini-1.5-flash-latest"
    # GOOGLE_EMBEDDING_MODEL_NAME = "models/embedding-001"
    ```
    **Note:** Replace `YOUR_GOOGLE_AI_STUDIO_API_KEY` with your actual API key obtained from Google AI Studio.

5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
    The application will typically open at `http://localhost:8501`.

### 🛠️ How It Works

1.  The user starts a new chat session or selects an existing one.
2.  For the selected session, the user uploads one or more PDF files and clicks "Process."
3.  The application extracts text from the PDFs and splits it into smaller chunks. These chunks are stored for both RAG and answering general questions.
4.  These text chunks are converted into vector embeddings using Google's embedding model.
5.  These vectors are stored in a FAISS vector store, specific to the active session, for efficient similarity searches.
6.  When the user asks a question:
    *   **If it's a General Question:** If the user's query contains general phrasing (e.g., "What is this PDF about?"), a portion of the stored text chunks and a special prompt template for structured output (summary or sample questions) are used to send a request to the Google Gemini LLM. The response is parsed using PydanticOutputParser and presented to the user.
    *   **If it's a Specific Question (RAG):** The text chunks most similar to the question's vector (relevant context) are retrieved from the FAISS vector store. This context and the user's question are formatted using a constrained RAG prompt template that guides the LLM to use only the provided information. The formatted prompt is sent to the Google Gemini LLM, and the response is displayed.
7.  Each chat session maintains its own PDF information, vector store, and chat history separately.

---

### ⚙️ `requirements.txt` İçeriği / Content

```txt
streamlit
pypdf2
langchain
langchain-google-genai
faiss-cpu
tiktoken
numpy<2.0
pydantic
uuid
langchain_community
