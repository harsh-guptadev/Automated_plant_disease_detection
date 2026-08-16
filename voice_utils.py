"""
voice_utils.py
==============
Voice input (Speech-to-Text) and voice output (Text-to-Speech) utilities
for the AgriVision AI chatbot.

Key fix: Each recording from mic_recorder has a unique `id`. We track the
last-processed ID in st.session_state so that on every Streamlit rerun the
same recording is NOT re-submitted — this allows follow-up questions to work
normally via both text typing and repeated voice recordings.
"""

import io
import base64
import re
import streamlit as st


# ---------------------------------------------------------------------------
# Language code mapping
# ---------------------------------------------------------------------------
LANG_CODE_MAP = {
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


# ---------------------------------------------------------------------------
# Speech-to-Text via streamlit-mic-recorder + SpeechRecognition
# ---------------------------------------------------------------------------

def render_voice_input(language: str = "English") -> str | None:
    """
    Renders a microphone recorder widget.

    Returns the transcribed text ONLY for a brand-new recording (identified
    by its unique `id`). On subsequent reruns the same recording is ignored,
    so normal text follow-up questions work without interference.
    """
    try:
        from streamlit_mic_recorder import mic_recorder  # type: ignore
    except ImportError:
        st.warning("Voice input unavailable: install `streamlit-mic-recorder`.")
        return None

    lang_code = LANG_CODE_MAP.get(language, "en-US")

    st.markdown(
        "<div style='margin-bottom:4px; font-size:0.9rem; color:#94a3b8;'>"
        "🎤 <b>Voice Input</b> — press <b>Start</b>, speak, then press <b>Stop</b>."
        "</div>",
        unsafe_allow_html=True,
    )

    # Dynamic key: changes after every exchange so the widget gets a fresh
    # Streamlit component instance — prevents state accumulation that causes
    # the mic to freeze after 3-4 responses.
    chat_len = len(st.session_state.get("chat_history", []))
    mic_key = f"agrivision_mic_{chat_len}"

    audio = mic_recorder(
        start_prompt="🎤  Start Speaking",
        stop_prompt="⏹  Stop & Transcribe",
        just_once=False,
        use_container_width=True,
        key=mic_key,
    )

    if audio is None:
        return None

    # ── Deduplicate within the same widget instance ────────────────────────
    audio_id = audio.get("id", -1)
    dedup_key = f"_last_mic_id_{mic_key}"
    if st.session_state.get(dedup_key) == audio_id:
        # Same recording already sent — do not resend
        return None
    st.session_state[dedup_key] = audio_id
    # ───────────────────────────────────────────────────────────────────────

    audio_bytes: bytes = audio.get("bytes", b"")
    if not audio_bytes:
        return None

    # Transcribe — convert WebM/Opus → WAV first (browser records WebM)
    try:
        import speech_recognition as sr  # type: ignore
        from pydub import AudioSegment   # type: ignore
    except ImportError:
        st.warning("Voice input unavailable: install `SpeechRecognition` and `pydub`.")
        return None

    recognizer = sr.Recognizer()
    # Keep energy threshold lower so quiet speech is detected
    recognizer.energy_threshold = 200
    recognizer.dynamic_energy_threshold = True

    with st.spinner("🔄 Transcribing your voice..."):
        try:
            webm_buffer = io.BytesIO(audio_bytes)
            audio_seg   = AudioSegment.from_file(webm_buffer)

            # Normalize volume and convert to 16 kHz mono (optimal for Google STT)
            target_dBFS   = -14.0
            audio_seg     = audio_seg.apply_gain(target_dBFS - audio_seg.dBFS)
            audio_seg     = audio_seg.set_frame_rate(16000).set_channels(1)

            wav_buffer = io.BytesIO()
            audio_seg.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            # NOTE: do NOT call adjust_for_ambient_noise() on AudioFile —
            # it is for live mic streams and silently eats the first N
            # seconds of pre-recorded audio, discarding speech.
            with sr.AudioFile(wav_buffer) as source:
                audio_data = recognizer.record(source)

            transcript = recognizer.recognize_google(
                audio_data, language=lang_code, show_all=False
            )
            st.success(f"🗣️ Heard: **{transcript}**")
            return transcript

        except sr.UnknownValueError:
            st.warning("⚠️ Could not understand — speak clearly and avoid background noise.")
            return None
        except sr.RequestError as e:
            st.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            st.error(f"Transcription error: {e}")
            return None


# ---------------------------------------------------------------------------
# Text-to-Speech via gTTS
# ---------------------------------------------------------------------------

def speak_response(text: str, language: str = "English") -> None:
    """
    Converts `text` to speech and embeds an HTML <audio> player in Streamlit.
    """
    try:
        from gtts import gTTS  # type: ignore
    except ImportError:
        st.caption("Install `gtts` to enable voice responses.")
        return

    lang_code = GTTS_LANG_MAP.get(language, "en")

    # Strip markdown symbols for cleaner TTS audio
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
