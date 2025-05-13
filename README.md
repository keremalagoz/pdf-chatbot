# 📚 Çoklu Sohbet PDF Asistanı / Multi-Session PDF Chat Assistant 📄

Bu Streamlit uygulaması, kullanıcıların PDF dosyaları yükleyerek bu dosyaların içeriği hakkında sorular sormasına olanak tanır. Uygulama, her bir PDF seti için ayrı sohbet oturumları oluşturur ve yönetir, böylece kullanıcılar farklı belgelerle ilgili sohbetlerini ayrı ayrı tutabilirler.

This Streamlit application allows users to upload PDF files and ask questions about their content. The application creates and manages separate chat sessions for each set of PDFs, enabling users to keep their conversations related to different documents separate.

---

## 🇹🇷 Türkçe Açıklama

### ✨ Özellikler

*   **PDF Yükleme:** Birden fazla PDF dosyası yüklenebilir.
*   **İçerik Tabanlı Soru Cevaplama:** Yüklenen PDF'lerin içeriğine dayalı olarak sorulara yanıt verir.
*   **Çoklu Sohbet Oturumları:** Her PDF seti veya sorgu için ayrı sohbet oturumları oluşturulabilir ve yönetilebilir.
*   **Yerel Embedding Modeli:** Metinleri vektörlere dönüştürmek için yerel bir `sentence-transformers` modeli kullanır (örn: `all-MiniLM-L6-v2`). Bu, embedding işlemi için API anahtarı gerektirmez ve verileriniz bu aşamada dışarı çıkmaz.
*   **OpenRouter Entegrasyonu:** Büyük Dil Modeli (LLM) yanıtları için OpenRouter.ai platformu üzerinden çeşitli (ücretsiz veya ücretli) modellere erişim sağlar.
*   **Sohbet Geçmişi:** Her oturum için sohbet geçmişi tutulur.
*   **Kısıtlayıcı Prompting:** LLM'in sadece yüklenen PDF içeriğine odaklanmasını sağlamak ve dışarıdan bilgi kullanmasını engellemek için özel olarak tasarlanmış prompt şablonu kullanılır.

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

4.  **API Anahtarlarını Ayarlayın (Streamlit Secrets):**
    Projenizin ana dizininde `.streamlit` adında bir klasör oluşturun ve içine `secrets.toml` adında bir dosya ekleyin.
    ```toml
    # .streamlit/secrets.toml

    OPENROUTER_API_KEY = "sk-or-v1-SENIN_OPENROUTER_API_ANAHTARIN"

    # İsteğe bağlı: Kullanılacak LLM ve yerel embedding modellerini değiştirmek için
    # LLM_MODEL_NAME = "mistralai/mistral-7b-instruct:free"
    # LOCAL_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    ```
    **Not:** `SENIN_OPENROUTER_API_ANAHTARIN` kısmını kendi OpenRouter API anahtarınızla değiştirin. OpenRouter.ai sitesinden ücretsiz bir hesap oluşturup API anahtarı alabilirsiniz.

5.  **Uygulamayı Çalıştırın:**
    ```bash
    streamlit run app.py
    ```
    Uygulama varsayılan olarak `http://localhost:8501` adresinde açılacaktır.

### 🛠️ Nasıl Çalışır?

1.  Kullanıcı bir veya daha fazla PDF dosyası yükler.
2.  Uygulama, PDF'lerden metinleri çıkarır ve daha küçük parçalara (chunks) böler.
3.  Yerel bir `sentence-transformers` modeli kullanılarak bu metin parçaları vektörlere (embeddings) dönüştürülür.
4.  Bu vektörler, hızlı benzerlik araması için bir FAISS vektör deposunda saklanır.
5.  Kullanıcı bir soru sorduğunda:
    *   Sorunun vektörüne en yakın olan metin parçaları (ilgili bağlam) vektör deposundan alınır.
    *   Bu bağlam ve kullanıcının sorusu, LLM'i sadece sağlanan bilgiyi kullanmaya yönlendiren özel bir prompt şablonu kullanılarak formatlanır.
    *   Formatlanmış prompt, OpenRouter üzerinden seçilen LLM'e gönderilir.
    *   LLM'den gelen yanıt kullanıcıya gösterilir.
6.  Her PDF seti için (veya başlatılan her yeni sohbet için) ayrı sohbet oturumları tutulur ve yönetilir.

---

## 🇬🇧🇺🇸 English Description

### ✨ Features

*   **PDF Upload:** Allows uploading multiple PDF files.
*   **Content-Based Q&A:** Answers questions based on the content of the uploaded PDFs.
*   **Multi-Session Chat:** Create and manage separate chat sessions for each set of PDFs or queries.
*   **Local Embedding Model:** Uses a local `sentence-transformers` model (e.g., `all-MiniLM-L6-v2`) to convert text into vectors. This does not require an API key for the embedding process, and your data stays local during this step.
*   **OpenRouter Integration:** Accesses various Large Language Models (LLMs) (free or paid) via the OpenRouter.ai platform for generating responses.
*   **Chat History:** Maintains chat history for each session.
*   **Constrained Prompting:** Utilizes a specifically designed prompt template to ensure the LLM focuses solely on the uploaded PDF content and prevents it from using external knowledge.

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

4.  **Set Up API Keys (Streamlit Secrets):**
    In your project's root directory, create a folder named `.streamlit` and add a file named `secrets.toml` inside it.
    ```toml
    # .streamlit/secrets.toml

    OPENROUTER_API_KEY = "sk-or-v1-YOUR_OPENROUTER_API_KEY"

    # Optional: To change the LLM and local embedding models used
    # LLM_MODEL_NAME = "mistralai/mistral-7b-instruct:free"
    # LOCAL_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    ```
    **Note:** Replace `YOUR_OPENROUTER_API_KEY` with your actual OpenRouter API key. You can get a free API key by signing up at OpenRouter.ai.

5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
    The application will typically open at `http://localhost:8501`.

### 🛠️ How It Works

1.  The user uploads one or more PDF files.
2.  The application extracts text from the PDFs and splits it into smaller chunks.
3.  These text chunks are converted into vector embeddings using a local `sentence-transformers` model.
4.  These vectors are stored in a FAISS vector store for efficient similarity searches.
5.  When the user asks a question:
    *   The text chunks most similar to the question's vector (relevant context) are retrieved from the vector store.
    *   This context and the user's question are formatted using a custom prompt template designed to guide the LLM to use only the provided information.
    *   The formatted prompt is sent to the selected LLM via OpenRouter.
    *   The response from the LLM is displayed to the user.
6.  Separate chat sessions are maintained and managed for each set of PDFs (or each new chat initiated).

---

### ⚙️ `requirements.txt` İçeriği / Content

```txt
streamlit
openai
pypdf
langchain
langchain-huggingface
sentence-transformers
faiss-cpu
tiktoken
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
numpy<2.0
uuid
