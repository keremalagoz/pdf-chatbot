import streamlit as st
import openai # OpenRouter için OpenAI kütüphanesini kullanıyoruz
# import os # Artık os.environ'a ihtiyacımız kalmadı, st.secrets her şeyi hallediyor

# --- Streamlit Secrets Kullanımı ---
# 1. Streamlit Cloud'a dağıtıyorsanız:
# Uygulamanızın Ayarlar (Settings) > Secrets bölümüne gidin ve aşağıdaki satırları ekleyin:
# OPENROUTER_API_KEY = "sk-or-v1-SENIN-OPENROUTER-API-ANAHTARIN"
# MODEL_NAME = "mistralai/mistral-7b-instruct:free" # (İsteğe bağlı, varsayılanı kullanabilirsiniz)

# 2. Yerel geliştirme yapıyorsanız:
# Proje ana dizininizde .streamlit/secrets.toml dosyası oluşturun ve içine:
# OPENROUTER_API_KEY = "sk-or-v1-SENIN-OPENROUTER-API-ANAHTARIN"
# MODEL_NAME = "mistralai/mistral-7b-instruct:free" # (İsteğe bağlı)
# satırlarını ekleyin.
# BU DOSYAYI .gitignore'A EKLEMEYİ UNUTMAYIN!

# Streamlit Secrets'tan API anahtarını ve model adını okuma
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
# Model adı için bir varsayılan değer belirleyelim, eğer Secrets'ta tanımlanmamışsa bu kullanılır.
MODEL_NAME = st.secrets.get("MODEL_NAME", "mistralai/mistral-7b-instruct:free")

# API anahtarı olmadan uygulama çalışamaz
if not OPENROUTER_API_KEY:
    st.error("OpenRouter API anahtarı bulunamadı! Lütfen Streamlit Secrets bölümüne 'OPENROUTER_API_KEY' olarak ekleyin.")
    st.caption("Yerelde çalışıyorsanız, projenizin `.streamlit/secrets.toml` dosyasına eklediğinizden emin olun.")
    st.stop() # Uygulamayı durdur

# OpenAI istemcisini OpenRouter için yapılandırma
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- Streamlit Arayüzü ---
st.set_page_config(page_title="OpenRouter Chatbot", page_icon="🤖")
st.title("🤖 OpenRouter Destekli Chatbot")
st.caption(f"Kullanılan Model: {MODEL_NAME}")

# Sohbet geçmişini session state'de saklama
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Sen yardımsever bir asistansın."} # Sistem mesajı
    ]

# Sohbet geçmişini gösterme
for message in st.session_state.messages:
    if message["role"] != "system": # Sistem mesajlarını arayüzde göstermeyelim
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Kullanıcıdan girdi alma
if prompt := st.chat_input("Mesajınızı yazın..."):
    # Kullanıcının mesajını geçmişe ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot'un yanıtını alma
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            messages_to_send = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_to_send,
                stream=True,
                # İsteğe bağlı: OpenRouter'a özel başlıklar
                # Streamlit Secrets'tan da alabilirsiniz:
                # http_referer = st.secrets.get("YOUR_SITE_URL")
                # x_title = st.secrets.get("YOUR_APP_NAME")
                # extra_headers={
                # "HTTP-Referer": http_referer if http_referer else "http://localhost:8501",
                # "X-Title": x_title if x_title else "Streamlit OpenRouter Chatbot"
                # }
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except openai.APIError as e:
            st.error(f"OpenRouter API Hatası: {e}")
            st.error(f"Detay: {e.body}") # Hata hakkında daha fazla bilgi için
            full_response = "Üzgünüm, API ile iletişimde bir sorun oluştu."
        except Exception as e:
            st.error(f"Beklenmedik bir hata oluştu: {e}")
            full_response = "Üzgünüm, bir hata oluştu."

    # Bot'un yanıtını geçmişe ekle
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sohbeti temizleme butonu
if st.sidebar.button("Sohbeti Temizle"):
    st.session_state.messages = [
        {"role": "system", "content": "Sen yardımsever bir asistansın."}
    ]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("OpenRouter.ai ve Streamlit ile güçlendirilmiştir.")
