"""
voice_utils.py
==============
Voice input (Speech-to-Text) and voice output (Text-to-Speech) utilities
for the AgriVision AI chatbot.

Speech-to-Text  : Uses the browser's native Web Speech API (no server-side
                  dependency, works on Chrome / Edge / Safari).
Text-to-Speech  : Uses gTTS (Google Text-to-Speech) to convert the assistant
                  response into an MP3 audio file that is streamed back in the
                  Streamlit app.
"""

import io
import base64
import tempfile
import streamlit as st
import streamlit.components.v1 as components


# ---------------------------------------------------------------------------
# Language code mapping (gTTS accepts IETF language tags)
# ---------------------------------------------------------------------------
LANG_CODE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
}

# ---------------------------------------------------------------------------
# Speech-to-Text  — Browser Web Speech API component
# ---------------------------------------------------------------------------

_STT_HTML = """
<style>
  .voice-wrapper {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
  }}
  #mic-btn {{
    background: {btn_bg};
    color: #ffffff;
    border: none;
    border-radius: 50px;
    padding: 10px 22px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 4px 14px rgba(52, 211, 153, 0.3);
  }}
  #mic-btn:hover {{
    transform: scale(1.04);
    box-shadow: 0 6px 20px rgba(52, 211, 153, 0.5);
  }}
  #mic-btn.listening {{
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    animation: pulse 1.2s infinite;
  }}
  @keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(239,68,68, 0.5); }}
    70%  {{ box-shadow: 0 0 0 10px rgba(239,68,68, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(239,68,68, 0); }}
  }}
  #transcript-box {{
    flex: 1;
    padding: 8px 14px;
    border-radius: 10px;
    background: rgba(15, 33, 26, 0.6);
    border: 1px solid rgba(52, 211, 153, 0.25);
    color: #e2e8f0;
    font-size: 0.9rem;
    min-height: 38px;
    word-break: break-word;
  }}
  #send-btn {{
    background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
    color: #0b1d16;
    border: none;
    border-radius: 50px;
    padding: 10px 18px;
    font-size: 0.85rem;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.15s ease;
  }}
  #send-btn:hover {{ transform: scale(1.05); }}
  #send-btn:disabled {{ opacity: 0.4; cursor: default; }}
  #voice-status {{
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 4px;
  }}
</style>

<div class="voice-wrapper">
  <button id="mic-btn" onclick="toggleListen()">🎤 {speak_label}</button>
  <div id="transcript-box">{placeholder}</div>
  <button id="send-btn" onclick="sendTranscript()" disabled>{send_label}</button>
</div>
<div id="voice-status">{status_idle}</div>

<script>
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition;
  let currentTranscript = "";
  let isListening = false;

  function toggleListen() {{
    if (!SpeechRecognition) {{
      document.getElementById("voice-status").innerText = "{no_support}";
      return;
    }}
    if (isListening) {{
      recognition.stop();
    }} else {{
      startListening();
    }}
  }}

  function startListening() {{
    recognition = new SpeechRecognition();
    recognition.lang = "{lang_code}";
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = function() {{
      isListening = true;
      document.getElementById("mic-btn").classList.add("listening");
      document.getElementById("mic-btn").innerText = "⏹ {stop_label}";
      document.getElementById("voice-status").innerText = "{status_listening}";
    }};

    recognition.onresult = function(event) {{
      let interim = "";
      let final_ = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {{
        if (event.results[i].isFinal) {{
          final_ += event.results[i][0].transcript;
        }} else {{
          interim += event.results[i][0].transcript;
        }}
      }}
      currentTranscript = final_ || interim;
      document.getElementById("transcript-box").innerText = currentTranscript;
      document.getElementById("send-btn").disabled = currentTranscript.trim() === "";
    }};

    recognition.onerror = function(event) {{
      document.getElementById("voice-status").innerText = "Error: " + event.error;
      stopListening();
    }};

    recognition.onend = function() {{
      stopListening();
    }};

    recognition.start();
  }}

  function stopListening() {{
    isListening = false;
    if (recognition) recognition.stop();
    document.getElementById("mic-btn").classList.remove("listening");
    document.getElementById("mic-btn").innerHTML = "🎤 {speak_label}";
    document.getElementById("voice-status").innerText = currentTranscript
      ? "{status_done}"
      : "{status_idle}";
  }}

  function sendTranscript() {{
    if (!currentTranscript.trim()) return;
    // Push transcript into Streamlit via query string approach
    const iframe = window.frameElement;
    const encoded = encodeURIComponent(currentTranscript.trim());
    const url = new URL(window.parent.location.href);
    url.searchParams.set("voice_input", encoded);
    window.parent.history.replaceState(null, "", url.toString());
    // Streamlit component value callback
    window.parent.postMessage({{type:"streamlit:componentReady", apiVersion: 1}}, "*");
    Streamlit.setComponentValue(currentTranscript.trim());
  }}
</script>
"""


def render_voice_input_component(language: str = "English", height: int = 120) -> str | None:
    """
    Renders a browser-based Speech-to-Text widget.
    Returns the transcribed text if the user clicked 'Send', else None.
    """
    lang_code = LANG_CODE_MAP.get(language, "en")

    labels = {
        "en": {
            "speak_label": "Speak",
            "stop_label": "Stop Listening",
            "send_label": "✉️ Send",
            "placeholder": "Your speech will appear here…",
            "status_idle": "Click 🎤 to start speaking",
            "status_listening": "🔴 Listening… Speak now",
            "status_done": "✅ Ready to send",
            "no_support": "⚠️ Your browser does not support voice input. Please use Chrome or Edge.",
        },
        "hi": {
            "speak_label": "बोलें",
            "stop_label": "रोकें",
            "send_label": "✉️ भेजें",
            "placeholder": "आपकी बात यहाँ दिखेगी…",
            "status_idle": "🎤 बोलने के लिए क्लिक करें",
            "status_listening": "🔴 सुन रहा हूँ…",
            "status_done": "✅ भेजने के लिए तैयार",
            "no_support": "⚠️ आपका ब्राउज़र वॉयस इनपुट का समर्थन नहीं करता।",
        },
        "es": {
            "speak_label": "Hablar",
            "stop_label": "Detener",
            "send_label": "✉️ Enviar",
            "placeholder": "Su voz aparecerá aquí…",
            "status_idle": "Haga clic en 🎤 para hablar",
            "status_listening": "🔴 Escuchando…",
            "status_done": "✅ Listo para enviar",
            "no_support": "⚠️ Su navegador no admite entrada de voz.",
        },
        "fr": {
            "speak_label": "Parler",
            "stop_label": "Arrêter",
            "send_label": "✉️ Envoyer",
            "placeholder": "Votre discours apparaîtra ici…",
            "status_idle": "Cliquez sur 🎤 pour parler",
            "status_listening": "🔴 À l'écoute…",
            "status_done": "✅ Prêt à envoyer",
            "no_support": "⚠️ Votre navigateur ne prend pas en charge la saisie vocale.",
        },
    }
    lbl = labels.get(lang_code, labels["en"])

    html = _STT_HTML.format(
        btn_bg="linear-gradient(135deg, #34d399 0%, #10b981 100%)",
        lang_code=lang_code,
        **lbl,
    )
    return components.html(html, height=height)


# ---------------------------------------------------------------------------
# Text-to-Speech  — gTTS output
# ---------------------------------------------------------------------------

def speak_response(text: str, language: str = "English") -> None:
    """
    Converts `text` to speech and embeds an HTML <audio> player in the
    Streamlit app so the user can listen to the assistant's reply.
    """
    try:
        from gtts import gTTS  # lazy import — gTTS may not be installed
    except ImportError:
        st.info("Install `gtts` in requirements.txt to enable voice responses.")
        return

    lang_code = LANG_CODE_MAP.get(language, "en")

    # Strip markdown-like formatting for cleaner TTS
    import re
    clean_text = re.sub(r"[*_`#>\[\]()~|]+", "", text)
    clean_text = re.sub(r"\n{2,}", ". ", clean_text).strip()
    if not clean_text:
        return

    try:
        tts = gTTS(text=clean_text[:2000], lang=lang_code, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode()

        audio_html = f"""
        <div style="margin-top:10px;">
          <audio controls autoplay style="width:100%; border-radius:10px; outline:none;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
          </audio>
          <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">
            🔊 AI Voice Response
          </div>
        </div>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.caption(f"Voice output unavailable: {e}")
