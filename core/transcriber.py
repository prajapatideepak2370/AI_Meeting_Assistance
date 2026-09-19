import logging
import os
import time
from pathlib import Path

import requests
import whisper
from dotenv import load_dotenv
from pydub import AudioSegment


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_PIECE_SECONDS = min(int(os.getenv("SARVAM_PIECE_SECONDS", "25")), 30)

_model = None
LOGGER = logging.getLogger(__name__)

def load_model():

    global _model

    if _model is None:
        LOGGER.info("Whisper model loading started: %s", WHISPER_MODEL)
        _model = whisper.load_model(WHISPER_MODEL)
        LOGGER.info("Whisper model loaded")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:

    model = load_model()

    LOGGER.info("Whisper transcription started: %s", chunk_path)
    started = time.perf_counter()
    result = model.transcribe(
        chunk_path,
        task="transcribe",
        fp16=False,
        verbose=False,
        condition_on_previous_text=False,
    )
    LOGGER.info("Whisper transcription completed in %.1fs: %s", time.perf_counter() - started, chunk_path)
    return result.get("text", "").strip()

def _send_to_sarvam(piece_path: str) -> str:
    """Send one 30-second-or-shorter WAV file to Sarvam."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in the .env file")

    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {
            "model": SARVAM_MODEL,
            "mode": "translate",
            "language_code": "unknown",
        }
        response = requests.post(
            SARVAM_STT_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        raise RuntimeError(
            f"Sarvam request failed ({response.status_code}): {response.text}"
        )

    transcript = response.json().get("transcript")
    if not isinstance(transcript, str):
        raise RuntimeError(f"Unexpected Sarvam response: {response.text}")
    return transcript.strip()

def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    transcript_parts = []
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  Sarvam piece {i + 1}/{total_pieces} ...")
            transcript_parts.append(_send_to_sarvam(piece_path))
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(transcript_parts).strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)

def transcribe_all(chunks: list, language: str = "english", progress_cb=None) -> str:

    if not chunks:
        raise ValueError("No audio chunks were created")

    full_transcript = []

    engine = "Sarvam AI" if language.lower() in {"hinglish", "sarvam"} else "Whisper"
    LOGGER.info("Using %s for transcription", engine)

    for i, chunk in enumerate(chunks):

        message = f"Transcribing chunk {i + 1}/{len(chunks)}..."
        LOGGER.info(message)
        if progress_cb:
            progress_cb(message, 30 + int(25 * i / len(chunks)))

        text = transcribe_chunk(chunk, language=language)

        if text:
            full_transcript.append(text)

    transcript = " ".join(full_transcript).strip()
    LOGGER.info("Transcription completed; transcript length: %s", len(transcript))
    if progress_cb:
        progress_cb("Transcription completed", 55)

    if not transcript:
        raise ValueError("Whisper returned an empty transcript")
    return transcript