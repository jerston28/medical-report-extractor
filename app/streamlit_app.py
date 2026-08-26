"""
Streamlit UI tying together the full pipeline:
ingestion -> classification -> extraction -> normalization -> summarization.

Upload a PDF or image medical report and see raw text, structured
fields, a plain-language summary, and flagged/abnormal items.
"""

import dataclasses
import io
import os
import sys
import tempfile

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_parser import is_text_based_pdf, extract_text_from_pdf, extract_images_from_pdf
from ingestion.ocr import ocr_image, ocr_pdf_pages
from ingestion.preprocess import preprocess_image
from classification.doc_type_classifier import classify_document
from extraction.lab_extractor import extract_lab_values
from extraction.prescription_extractor import extract_prescription_info
from extraction.discharge_extractor import extract_discharge_info
from normalization.normalize import normalize_extraction
from summarization.llm_summarizer import summarize_report

st.set_page_config(page_title="Medical Report Extractor", layout="wide")

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

EXTRACTORS = {
    "lab": extract_lab_values,
    "prescription": extract_prescription_info,
    "discharge": extract_discharge_info,
}


def get_raw_text_from_pdf(pdf_path: str) -> str:
    if is_text_based_pdf(pdf_path):
        return extract_text_from_pdf(pdf_path)
    images = extract_images_from_pdf(pdf_path)
    images = [preprocess_image(img) for img in images]
    return ocr_pdf_pages(images)


def get_raw_text_from_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes))
    image = preprocess_image(image)
    return ocr_image(image)


def to_jsonable(normalized):
    if isinstance(normalized, list):
        return [dataclasses.asdict(item) for item in normalized]
    return dataclasses.asdict(normalized)


def get_flagged_items(doc_type: str, normalized):
    """
    Pull out whatever counts as "flagged/abnormal" for this doc type.
    Lab results have an explicit flag field; prescriptions and discharge
    summaries don't have a normal/abnormal concept in the structured
    data, so surface the clinically-relevant fields instead rather than
    fabricating a flag that isn't there.

    Defensive by design: extraction can legitimately come back empty or
    partial (an unfamiliar section heading, a garbled OCR page, a
    format the extractor wasn't built for), so this never assumes a
    field is populated or even present - it always degrades to an
    empty list/None rather than raising, for all three doc types.
    """
    if doc_type == "lab":
        if not normalized:
            return []
        return [
            dataclasses.asdict(row) for row in normalized
            if getattr(row, "flag", None) and row.flag.strip().lower() not in ("normal", "")
        ]
    if doc_type == "discharge":
        if not normalized:
            return []
        entities = getattr(normalized, "diagnosis_entities", None) or []
        canonical_diagnosis = getattr(normalized, "canonical_diagnosis", None)
        if canonical_diagnosis:
            return [{"diagnosis": canonical_diagnosis, "entities": entities}]
        return []
    if doc_type == "prescription":
        return None  # prescriptions have no structured flag concept
    return None


@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes: bytes, extension: str):
    """
    Runs the full pipeline on uploaded file bytes. Cached on file
    content so re-rendering the page (e.g. switching tabs) doesn't
    re-run OCR or re-call the summarization API every time.
    Returns a dict with either the results or an "error" key.
    """
    try:
        if extension in PDF_EXTENSIONS:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                raw_text = get_raw_text_from_pdf(tmp_path)
            finally:
                # Best-effort cleanup only: on Windows, a corrupted/invalid
                # PDF can leave MuPDF holding a file lock even after
                # fitz.open() raises, which makes os.remove() itself raise
                # WinError 32. If that's allowed to propagate here, it
                # replaces (masks) the real, useful error from above with a
                # confusing "file in use" message. A leftover temp file is
                # harmless (OS temp dirs get cleaned periodically); silently
                # masking the actual failure reason is not.
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        else:
            raw_text = get_raw_text_from_image(file_bytes)
    except Exception as e:
        return {"error": f"Failed to read/OCR this file: {e}"}

    if not raw_text or not raw_text.strip():
        return {"error": "No readable text was found in this file. The scan quality may be too poor, or the file may be blank."}

    try:
        doc_type = classify_document(raw_text)
        raw_extraction = EXTRACTORS[doc_type](raw_text)
        normalized = normalize_extraction(doc_type, raw_extraction)
    except Exception as e:
        return {"error": f"Extraction/normalization failed: {e}", "raw_text": raw_text}

    summary = summarize_report(normalized, doc_type)

    return {
        "raw_text": raw_text,
        "doc_type": doc_type,
        "raw_extraction": raw_extraction,
        "normalized": normalized,
        "summary": summary,
    }


def render_extracted_fields(doc_type: str, normalized):
    if doc_type in ("lab", "prescription"):
        rows = to_jsonable(normalized)
        if not rows:
            st.info("No structured fields were extracted from this document.")
        else:
            st.dataframe(rows, use_container_width=True)
    else:
        st.json(to_jsonable(normalized))


def render_flagged(doc_type: str, normalized):
    flagged = get_flagged_items(doc_type, normalized)
    if flagged is None:
        st.info("Prescriptions don't have a normal/abnormal flag in the structured data - see the Plain-Language Summary tab for medication-specific notes (e.g. unclear dosing instructions).")
    elif not flagged:
        st.success("No abnormal or flagged items found.")
    else:
        st.json(flagged)


def main():
    st.title("Medical Report Information Extraction & Summarization")
    st.caption("Upload a lab report, prescription, or discharge summary (PDF or image) to see it extracted, normalized, and summarized in plain language.")

    uploaded_file = st.file_uploader("Upload a report", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is None:
        st.info("Upload a file to get started. Sample files are available under data/samples/.")
        return

    extension = os.path.splitext(uploaded_file.name)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        st.error(f"Unsupported file type: {extension}. Please upload a PDF, PNG, or JPG.")
        return

    file_bytes = uploaded_file.getvalue()

    with st.spinner("Running ingestion, extraction, normalization, and summarization..."):
        result = run_pipeline(file_bytes, extension)

    if "error" in result:
        st.error(result["error"])
        if "raw_text" in result:
            with st.expander("Raw extracted text (extraction/normalization failed after this point)"):
                st.text(result["raw_text"])
        return

    st.success(f"Detected document type: **{result['doc_type']}**")

    tab_raw, tab_fields, tab_summary, tab_flagged = st.tabs(
        ["Raw Text", "Extracted Fields", "Plain-Language Summary", "Flagged/Abnormal Items"]
    )

    with tab_raw:
        st.text_area("Raw text from ingestion", result["raw_text"], height=400)

    with tab_fields:
        render_extracted_fields(result["doc_type"], result["normalized"])

    with tab_summary:
        if result["summary"].startswith("Error:"):
            st.error(result["summary"])
        else:
            st.markdown(result["summary"])

    with tab_flagged:
        render_flagged(result["doc_type"], result["normalized"])


if __name__ == "__main__":
    main()
