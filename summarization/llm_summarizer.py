"""
Calls the Groq API to turn normalized structured data into a
plain-language patient summary.
"""

import os

from dotenv import load_dotenv
from groq import Groq

from summarization.prompts import SYSTEM_PROMPT, build_summary_prompt

load_dotenv()

MODEL = "openai/gpt-oss-20b"

# openai/gpt-oss-20b is a reasoning model - it spends part of its token
# budget "thinking" before writing the final answer, so max_tokens needs
# real headroom or the response comes back empty.
MAX_TOKENS = 900


def summarize_report(normalized_data, doc_type: str) -> str:
    """
    Build the prompt for `normalized_data`/`doc_type` and call Groq to
    generate a plain-language summary. Returns the summary text, or a
    human-readable "Error: ..." string if the call fails for any reason
    (missing API key, network/auth failure, empty response) - callers
    (the Streamlit app) can display this directly rather than crashing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY is not set. Add it to your .env file to enable summarization."

    try:
        prompt = build_summary_prompt(normalized_data, doc_type)
    except ValueError as e:
        return f"Error: {e}"

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.3,
        )
    except Exception as e:
        return f"Error: Could not reach the summarization service ({e})."

    if not response.choices:
        return "Error: The summarization service returned no response."

    content = response.choices[0].message.content
    if not content or not content.strip():
        return "Error: The summarization service returned an empty summary. Please try again."

    return content.strip()
