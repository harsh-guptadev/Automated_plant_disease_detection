"""
voice_utils.py
==============
Voice input (Speech-to-Text) and voice output (Text-to-Speech) utilities
for the AgriVision AI chatbot.

STT Strategy:
  Primary   → openai/whisper-large-v3 via Hugging Face Inference API
              (excellent multilingual support — Hindi, Spanish, French, English)
  Fallback  → Google Speech API (English-only fallback, no key required)

TTS:
  gTTS (Google Text-to-Speech)
"""

import io
import os
import base64
import re
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Language mapping
# ─────────────────────────────────────────────────────────────────────────────
WHISPER_LANG_MAP = {
    "English": "english",
    "Hindi":   "hindi",
    "Spanish": "spanish",
    "French":  "french",
}

GOOGLE_LANG_MAP = {
    "English": "en-US",
    "Hindi":   "hi-IN",
    "Spanish": "es-ES",
    "French":  "fr-FR",
}

GTTS_LANG_MAP = {
    "English": "en",
    "Hindi":   "hi",
    "Spanish": "es",
    "French":  "fr",
}

# HF Whisper endpoints in priority order
WHISPER_MODELS = [
    "openai/whisper-large-v3",
    "openai/whisper-medium",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _webm_to_wav(audio_bytes: bytes) -> bytes:
    """Convert browser WebM/Opus bytes to 16-kHz mono WAV using pydub."""
    try:
        from pydub import AudioSegment  # type: ignore
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg = seg.set_frame_rate(16000).set_channels(1)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        return buf.getvalue()
    except Exception:
        # If pydub/ffmpeg not available, return raw bytes and hope for the best
        return audio_bytes


def _transcribe_whisper(wav_bytes: bytes, language: str, hf_token: str) -> str | None:
    """
    Send WAV bytes to HF Whisper API with an explicit language parameter.
    Using JSON + base64 payload ensures the language hint is honoured by Whisper.
    """
    # Map to ISO 639-1 codes that Whisper accepts
    lang_iso = {
        "English": "en",
        "Hindi":   "hi",
        "Spanish": "es",
        "French":  "fr",
    }.get(language, "en")

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type":  "application/json",
    }
    # Base64-encode audio so it travels cleanly inside JSON
    audio_b64 = base64.b64encode(wav_bytes).decode()
    payload = {
        "inputs": audio_b64,
        "parameters": {"language": lang_iso},
    }

    for model in WHISPER_MODELS:
        url = f"https://api-inference.huggingface.co/models/{model}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=40)

            if resp.status_code == 200:
                data = resp.json()
                # HF returns {"text": "..."} for ASR
                text = data.get("text", "").strip() if isinstance(data, dict) else ""
                if text:
                    return text

            elif resp.status_code == 503:
                # Model still loading — wait a moment and retry once
                import time
                time.sleep(4)
                resp2 = requests.post(url, headers=headers, json=payload, timeout=40)
                if resp2.status_code == 200:
                    data = resp2.json()
                    text = data.get("text", "").strip() if isinstance(data, dict) else ""
                    if text:
                        return text

        except requests.Timeout:
            continue
        except Exception:
            continue

    return None


def _transcribe_google(wav_bytes: bytes, language: str) -> str | None:
    """Fallback: Google free STT (English works best)."""
    try:
        import speech_recognition as sr  # type: ignore
    except ImportError:
        return None

    lang_code  = GOOGLE_LANG_MAP.get(language, "en-US")
    recognizer = sr.Recognizer()
    recognizer.energy_threshold      = 200
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language=lang_code, show_all=False)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public: Speech-to-Text widget
# ─────────────────────────────────────────────────────────────────────────────

def render_voice_input(language: str = "English") -> str | None:
    """
    Renders a microphone recorder widget.
    Returns the transcribed text ONLY for a brand-new recording.
    On subsequent reruns the same recording is ignored so typed follow-up
    questions work without interference.
    """
    try:
        from streamlit_mic_recorder import mic_recorder  # type: ignore
    except ImportError:
        st.warning("Voice input unavailable: install `streamlit-mic-recorder`.")
        return None

    st.markdown(
        "<div style='margin-bottom:4px; font-size:0.9rem; color:#94a3b8;'>"
        "🎤 <b>Voice Input</b> — press <b>Start</b>, speak your question, then press <b>Stop</b>."
        "</div>",
        unsafe_allow_html=True,
    )

    # Dynamic key per exchange — prevents component state accumulation freeze
    chat_len = len(st.session_state.get("chat_history", []))
    mic_key  = f"agrivision_mic_{chat_len}"

    audio = mic_recorder(
        start_prompt="🎤  Start Speaking",
        stop_prompt="⏹  Stop & Transcribe",
        just_once=False,
        use_container_width=True,
        key=mic_key,
    )

    if audio is None:
        return None

    # Deduplicate — only process each recording once
    audio_id  = audio.get("id", -1)
    dedup_key = f"_last_mic_id_{mic_key}"
    if st.session_state.get(dedup_key) == audio_id:
        return None
    st.session_state[dedup_key] = audio_id

    audio_bytes: bytes = audio.get("bytes", b"")
    if not audio_bytes:
        return None

    with st.spinner("🔄 Transcribing your voice..."):
        # Convert WebM → WAV once
        wav_bytes = _webm_to_wav(audio_bytes)

        # Try Whisper (HF API) first — best multilingual support
        hf_token = (
            st.session_state.get("HF_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )

        transcript = None
        if hf_token:
            transcript = _transcribe_whisper(wav_bytes, language, hf_token)

        # Fallback to Google STT
        if not transcript:
            transcript = _transcribe_google(wav_bytes, language)

        if transcript:
            st.success(f"🗣️ Heard: **{transcript}**")
            return transcript
        else:
            st.warning(
                "⚠️ Could not understand your speech. "
                "Tips: speak clearly, hold mic close, reduce background noise."
            )
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Public: Text-to-Speech
# ─────────────────────────────────────────────────────────────────────────────

def speak_response(text: str, language: str = "English") -> None:
    """Convert text to speech and embed an HTML audio player in Streamlit."""
    try:
        from gtts import gTTS  # type: ignore
    except ImportError:
        st.caption("Install `gtts` to enable voice responses.")
        return

    lang_code = GTTS_LANG_MAP.get(language, "en")

    clean = re.sub(r"[*_`#>\[\]()~|]+", "", text)
    clean = re.sub(r"\n{2,}", ". ", clean).strip()
    if not clean:
        return

    try:
        tts = gTTS(text=clean[:2000], lang=lang_code, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()

        st.markdown(
            f"""
            <div style="margin-top:10px;">
              <audio controls autoplay
                     style="width:100%; border-radius:10px; outline:none;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
              </audio>
              <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">
                🔊 AI Voice Response
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.caption(f"Voice output unavailable: {e}")
