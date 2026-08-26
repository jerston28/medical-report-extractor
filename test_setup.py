"""
Stage 1 sanity check — run this after installing everything
to confirm your environment is ready for Stage 2.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def check_imports():
    print("Checking imports...")
    try:
        import fitz  # pymupdf
        print("  [OK] pymupdf")
    except ImportError as e:
        print(f"  [FAIL] pymupdf: {e}")

    try:
        import pytesseract
        print("  [OK] pytesseract")
    except ImportError as e:
        print(f"  [FAIL] pytesseract: {e}")

    try:
        import spacy
        print("  [OK] spacy")
    except ImportError as e:
        print(f"  [FAIL] spacy: {e}")

    try:
        import sklearn
        print("  [OK] scikit-learn")
    except ImportError as e:
        print(f"  [FAIL] scikit-learn: {e}")

    try:
        import streamlit
        print("  [OK] streamlit")
    except ImportError as e:
        print(f"  [FAIL] streamlit: {e}")

    try:
        import groq
        print("  [OK] groq")
    except ImportError as e:
        print(f"  [FAIL] groq: {e}")


def check_tesseract():
    print("\nChecking Tesseract binary...")
    import shutil
    import pytesseract

    # If tesseract is already on PATH, pytesseract will find it with no
    # config needed. Otherwise, fall back to common install locations.
    if shutil.which("tesseract") is None:
        candidates = [
            os.path.expanduser(r"~\Downloads\tesseract.exe"),      # confirmed local install
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",       # Windows default
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe", # Windows 32-bit default
            "/opt/homebrew/bin/tesseract",                          # macOS (Apple Silicon, Homebrew)
            "/usr/local/bin/tesseract",                             # macOS (Intel, Homebrew)
            "/usr/bin/tesseract",                                   # Linux
        ]
        found = next((p for p in candidates if os.path.isfile(p)), None)
        if found:
            pytesseract.pytesseract.tesseract_cmd = found
        else:
            # UNCOMMENT and edit this line with your actual install path
            # (use a raw string, e.g. r'C:\...', so backslashes aren't
            # misread as escape sequences like \t for tab):
            # pytesseract.pytesseract.tesseract_cmd = r'C:\path\to\tesseract.exe'
            pass

    try:
        version = pytesseract.get_tesseract_version()
        print(f"  [OK] Tesseract version: {version}")
        if pytesseract.pytesseract.tesseract_cmd != "tesseract":
            print(f"  -> Using: {pytesseract.pytesseract.tesseract_cmd}")
    except Exception as e:
        print(f"  [FAIL] Tesseract not found: {e}")
        print("  -> Set pytesseract.pytesseract.tesseract_cmd to your install path (as a raw string)")


def check_groq_key():
    print("\nChecking Groq API key...")
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "your_key_here":
        print("  [FAIL] GROQ_API_KEY not set in .env")
        return
    try:
        from groq import Groq
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "Say 'test ok' and nothing else."}],
            max_tokens=10,
        )
        print(f"  [OK] Groq responded: {response.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"  [FAIL] Groq call failed: {e}")


def check_scispacy_model():
    print("\nChecking scispaCy model...")
    try:
        import spacy
        nlp = spacy.load("en_core_sci_sm")
        doc = nlp("Patient prescribed 500mg Amoxicillin twice daily.")
        print(f"  [OK] Model loaded. Sample tokens: {[t.text for t in doc][:5]}")
    except Exception as e:
        print(f"  [FAIL] scispaCy model not loaded: {e}")


if __name__ == "__main__":
    check_imports()
    check_tesseract()
    check_groq_key()
    check_scispacy_model()
    print("\nDone. Fix any [FAIL] lines above before moving to Stage 2.")