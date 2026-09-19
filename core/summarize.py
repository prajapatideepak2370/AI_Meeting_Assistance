import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
LOGGER = logging.getLogger(__name__)

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured in the project .env file")
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=api_key,
        temperature=0.3,
        timeout=120,
        max_retries=2,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)

def summarize(transcript : str) -> str:
    if not transcript.strip():
        raise ValueError("Cannot summarize an empty transcript")
    LOGGER.info("Summary generation started; transcript length: %s", len(transcript))
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = []
    for index, chunk in enumerate(chunks, start=1):
        LOGGER.info("Summary API request sent for transcript section %s/%s", index, len(chunks))
        summary = map_chain.invoke({"text" : chunk})
        LOGGER.info("Summary API response received for transcript section %s/%s", index, len(chunks))
        chunk_summaries.append(summary)

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    LOGGER.info("Summary API request sent for final combination")
    result = combined_chain.invoke(combined)
    LOGGER.info("Summary generated")
    return result

def generate_title(transcipt : str) -> str:
    if not transcipt.strip():
        raise ValueError("Cannot generate a title from an empty transcript")
    llm = get_llm()

    

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    return title_chain.invoke(transcipt[:2000])
