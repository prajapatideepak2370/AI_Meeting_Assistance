import logging
import os
import shutil
from pathlib import Path

import yt_dlp
from pydub import AudioSegment

LOGGER = logging.getLogger(__name__)
DOWNLOAD_DIR = Path(__file__).resolve().parents[1] / "downloades"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def find_ffmpeg_location() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return os.path.dirname(ffmpeg_path)

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        for ffmpeg_path in package_root.glob("Gyan.FFmpeg.Shared*/**/bin/ffmpeg.exe"):
            return str(ffmpeg_path.parent)

    return None


def download_youtube_audio(url : str) -> str:
    output_path = str(DOWNLOAD_DIR / "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "noplaylist": True,
    }
    ffmpeg_location = find_ffmpeg_location()
    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location
    deno_path = shutil.which("deno")
    if deno_path:
        ydl_opts["js_runtimes"] = {"deno": {"path": deno_path}}

    LOGGER.info("YouTube audio download started")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = Path(ydl.prepare_filename(info)).with_suffix(".wav")
    if not filename.is_file():
        raise FileNotFoundError(f"Downloaded WAV was not created: {filename}")
    LOGGER.info("YouTube audio download completed: %s", filename)
    return str(filename)

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    LOGGER.info("Audio conversion started: %s", input_path)
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    LOGGER.info("Audio conversion completed: %s", output_path)
    return output_path

# chunking the audio 
def chunk_audio(wav_path : str , chunk_minutes : int | None = None) -> list:
    if chunk_minutes is None:
        chunk_minutes = max(1, int(os.getenv("AUDIO_CHUNK_MINUTES", "2")))
    if chunk_minutes < 1:
        raise ValueError("AUDIO_CHUNK_MINUTES must be at least 1")

    LOGGER.info("Audio chunking started: %s (%s minute chunks)", wav_path, chunk_minutes)
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    if not chunks:
        raise ValueError("Audio file contains no audio")
    LOGGER.info("Audio chunking completed: %s chunk(s)", len(chunks))
    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        LOGGER.info("Detected YouTube URL")
        wav_path = download_youtube_audio(source)
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Input file does not exist: {source}")
        LOGGER.info("Detected local file")
        wav_path = convert_to_wav(source)

    LOGGER.info("Chunking audio")
    chunks = chunk_audio(wav_path)
    LOGGER.info("Audio ready: %s chunk(s) created", len(chunks))
    return chunks