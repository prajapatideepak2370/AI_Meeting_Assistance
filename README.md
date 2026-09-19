🎥 AI Video Meeting Assistant

An AI-powered Video Meeting & Study Assistant that processes meeting
recordings, lectures, classes, seminars, interviews, and YouTube videos.
It can transcribe media, generate summaries, extract action items and
key decisions, identify open questions, and provide a RAG-powered chat
interface over the transcript.

✨ Features

YouTube URL processing with yt-dlp

Local audio/video upload

English transcription with OpenAI Whisper

Hinglish transcription/translation with Sarvam AI

Automatic meeting title generation

AI-generated meeting summaries

Action-item extraction with owner and deadline when mentioned

Key-decision extraction

Open-question extraction

RAG-based chat over the transcript

ChromaDB vector search with all-MiniLM-L6-v2

Transcript display and TXT download

Streamlit progress tracking

🧠 Architecture

YouTube URL / Local File
          ↓
yt-dlp / pydub + FFmpeg
          ↓
WAV Conversion (Mono, 16 kHz)
          ↓
Audio Chunking
          ↓
Whisper / Sarvam AI
          ↓
Transcript
     ┌────┼──────────────┐
     ↓    ↓              ↓
 Summary  Extraction    RAG Index
          ↓              ↓
 Actions, Decisions   ChromaDB
 Questions               ↓
                    Retriever
                        ↓
                    LLM Chat

🛠️ Tech Stack

Layer             Technologies

UI                Streamlit
Media             yt-dlp, FFmpeg, pydub
Speech-to-Text    OpenAI Whisper, Sarvam AI
LLM               LangChain, LangChain-Groq, Groq openai/gpt-oss-20b
RAG               ChromaDB, Sentence Transformers
Embeddings        all-MiniLM-L6-v2
Language/Config   Python, python-dotenv
Supporting        PyTorch, Torchaudio, NumPy, Requests, tqdm, tiktoken

📂 Project Structure

AI_Meeting_Assistance/
├── app.py
├── Requirements.txt
├── .gitignore
├── core/
│   ├── extractor.py
│   ├── rag_engine.py
│   ├── summarize.py
│   ├── transcriber.py
│   └── vector_store.py
└── utils/
    └── audio_processor.py

⚙️ Processing Pipeline

Select a YouTube URL or upload a local recording.

Download/extract audio or convert the uploaded media to WAV.

Convert audio to mono 16 kHz.

Split audio into 10-minute chunks.

Transcribe using Whisper for English or Sarvam AI for Hinglish.

Generate a professional meeting title.

Generate a concise bullet-point summary.

Extract action items, key decisions, and open questions.

Split the transcript into RAG chunks and create embeddings.

Store embeddings in ChromaDB.

Retrieve relevant transcript chunks for user questions.

Generate answers using the retrieved meeting context.

🚀 Installation

Prerequisites

Python 3.10+ recommended

FFmpeg installed and available on PATH

Groq API key

Sarvam API key for Hinglish mode

Clone

git clone https://github.com/prajapatideepak2370/AI_Meeting_Assistance.git
cd AI_Meeting_Assistance

Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate

macOS / Linux

python3 -m venv venv
source venv/bin/activate

Install Dependencies

pip install -r Requirements.txt

🔑 Environment Variables

Create .env in the project root:

GROQ_API_KEY=your_groq_api_key

# Required for Hinglish mode
SARVAM_API_KEY=your_sarvam_api_key

# Optional
WHISPER_MODEL=small
SARVAM_STT_MODEL=saaras:v3
SARVAM_PIECE_SECONDS=25

Never commit API keys or .env to GitHub.

▶️ Run

streamlit run app.py

🎬 Supported Upload Formats

The current Streamlit interface accepts:

MP4

MOV

MKV

MP3

WAV

M4A

WEBM

💬 RAG Chat

The transcript is split into chunks and embedded with
all-MiniLM-L6-v2. ChromaDB performs similarity retrieval, and the
retrieved transcript context is supplied to the Groq-hosted LLM.

The RAG prompt instructs the assistant to answer from the meeting
transcript context and to explicitly say when the requested information
cannot be found.

🔐 Configuration

Variable                 Required   Default       Purpose

GROQ_API_KEY           Yes        ---           LLM access
SARVAM_API_KEY         Hinglish   ---           Sarvam speech-to-text
WHISPER_MODEL          No         small       Whisper model
SARVAM_STT_MODEL       No         saaras:v3   Sarvam model
SARVAM_PIECE_SECONDS   No         25          Sarvam audio piece size

📌 Use Cases

Business meetings

College lectures

Online classes

Technical seminars

Interviews

Team discussions

Educational YouTube videos

Long-recording search and Q&A

⚠️ Notes

Long recordings can take several minutes to process.

Whisper may download/load a model on first use.

FFmpeg is required for media conversion.

Hinglish mode requires Sarvam API access.

The local vector store is created in vector_db.

Downloaded YouTube audio is stored in the downloades directory.

🔮 Future Improvements

Speaker diarization

Timestamp-aware answers

More multilingual support

PDF meeting reports

Persistent meeting history

Authentication

Separate vector collections per meeting

Better source citations in RAG responses

Docker deployment

👨‍💻 Author

Deepak Prajapati
