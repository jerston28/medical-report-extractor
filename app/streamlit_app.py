"""
Streamlit UI tying together the full pipeline:
ingestion -> classification -> extraction -> normalization -> summarization.

Upload a PDF or image medical report and see raw text, structured
fields, a plain-language summary, and flagged/abnormal items.
"""

import dataclasses
import html
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

st.set_page_config(page_title="Medical Report Extractor", page_icon="\U0001FA7A", layout="wide")

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

EXTRACTORS = {
    "lab": extract_lab_values,
    "prescription": extract_prescription_info,
    "discharge": extract_discharge_info,
}

DOC_TYPE_LABELS = {
    "lab": ("Lab Report", "\U0001F9EA"),
    "prescription": ("Prescription", "\U0001F48A"),
    "discharge": ("Discharge Summary", "\U0001F3E5"),
}

GITHUB_URL = "https://github.com/jerston28/medical-report-extractor"


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Hide the default Streamlit footer - we render our own below. */
        footer { visibility: hidden; }

        .app-header {
            padding: 0.25rem 0 1.25rem 0;
            border-bottom: 1px solid #E2E8F0;
            margin-bottom: 1.5rem;
        }
        .app-header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
            color: #0F172A;
        }
        .app-header .tagline {
            color: #475569;
            font-size: 1rem;
            margin: 0;
        }

        /* File uploader: rounded, teal-tinted dropzone with more visual weight */
        [data-testid="stFileUploaderDropzone"] {
            border-radius: 12px;
            border: 1.5px dashed #5EEAD4;
            background-color: #F0FDFA;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #0D9488;
        }

        /* Tabs: clearer active-tab distinction */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1.5px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 46px;
            border-radius: 8px 8px 0 0;
            padding: 0 18px;
            color: #64748B;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            color: #0D9488 !important;
            font-weight: 700 !important;
            border-bottom: 3px solid #0D9488 !important;
            background-color: #F0FDFA;
        }

        /* Detected document type badge */
        .doc-type-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: linear-gradient(90deg, #0D9488, #0891B2);
            color: white;
            padding: 0.45rem 1.1rem;
            border-radius: 999px;
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        /* Generic field card */
        .field-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #0D9488;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .field-card .field-title {
            font-weight: 700;
            font-size: 1.02rem;
            color: #0F172A;
            margin-bottom: 0.3rem;
        }
        .field-card .field-subtitle {
            color: #64748B;
            font-size: 0.85rem;
            margin-bottom: 0.4rem;
        }
        .field-card .field-row {
            font-size: 0.92rem;
            color: #334155;
            margin: 0.15rem 0;
        }
        .field-card .field-row b { color: #0F172A; }

        /* Flag/status pills */
        .status-pill {
            display: inline-block;
            padding: 0.15rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .status-normal { background-color: #ECFDF5; color: #047857; }
        .status-abnormal { background-color: #FEF2F2; color: #B91C1C; }
        .status-warning { background-color: #FFFBEB; color: #B45309; }
        .status-neutral { background-color: #F1F5F9; color: #475569; }

        /* Flagged-item cards lean warmer to draw attention */
        .flag-card {
            background-color: #FFFBFA;
            border: 1px solid #FCA5A5;
            border-left: 4px solid #DC2626;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.75rem;
        }
        .flag-card .field-title { color: #991B1B; }

        .entity-chip {
            display: inline-block;
            background-color: #EFF6FF;
            color: #1D4ED8;
            border-radius: 999px;
            padding: 0.15rem 0.6rem;
            font-size: 0.78rem;
            margin: 0.15rem 0.3rem 0.15rem 0;
        }

        .app-footer {
            margin-top: 3rem;
            padding-top: 1.25rem;
            border-top: 1px solid #E2E8F0;
            color: #94A3B8;
            font-size: 0.85rem;
            text-align: center;
        }
        .app-footer a { color: #0D9488; text-decoration: none; font-weight: 600; }
        .app-footer a:hover { text-decoration: underline; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="app-header">
            <h1>\U0001FA7A Medical Report Extractor</h1>
            <p class="tagline">Turn lab reports, prescriptions, and discharge summaries into structured data and plain-language summaries.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        f"""
        <div class="app-footer">
            All sample data is synthetic/demo data - no real patient records are used or stored.<br/>
            <a href="{GITHUB_URL}" target="_blank">View source on GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_class(flag: str) -> str:
    flag_lower = (flag or "").strip().lower()
    if flag_lower in ("normal", ""):
        return "status-normal"
    if flag_lower in ("high", "low", "abnormal", "critical"):
        return "status-abnormal"
    return "status-warning"


# ---------------------------------------------------------------------
# Pipeline (unchanged logic - presentation only lives below this point)
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def render_lab_cards(rows: list):
    for row in rows:
        d = dataclasses.asdict(row)
        status_class = _status_class(d.get("flag"))
        flag_label = html.escape((d.get("flag") or "Not flagged").strip() or "Not flagged")
        name = html.escape(d["canonical_test_name"] or d["test_name"])
        unit = html.escape(d.get("unit") or "")
        reference_range = html.escape(d.get("reference_range") or "not provided")
        # Single line - see render_prescription_cards for why.
        st.markdown(
            f'<div class="field-card"><div class="field-title">{name}'
            f'<span class="status-pill {status_class}" style="float:right;">{flag_label}</span></div>'
            f'<div class="field-row"><b>{d["value"]} {unit}</b></div>'
            f'<div class="field-row">Reference range: {reference_range}</div></div>',
            unsafe_allow_html=True,
        )


def render_prescription_cards(rows: list):
    # Built as a single concatenated line, not an indented multi-line
    # f-string: Streamlit's markdown renderer follows CommonMark, where a
    # blank line inside an HTML block ends that block, and any indented
    # line after it gets treated as a code block instead of raw HTML.
    # The "as written" subtitle is conditionally empty, which produced
    # exactly that blank line and caused the rest of the card to print
    # as literal HTML text instead of rendering - keeping everything on
    # one line sidesteps the issue entirely.
    for row in rows:
        d = dataclasses.asdict(row)
        name = html.escape(d.get("canonical_drug_name") or d.get("drug_name") or "Unknown medication")
        subtitle_html = ""
        if d.get("drug_name") and d["drug_name"].lower() != (d.get("canonical_drug_name") or "").lower():
            subtitle_html = f'<div class="field-subtitle">as written: {html.escape(d["drug_name"])}</div>'
        dosage = html.escape(d.get("dosage") or "not specified")
        frequency = html.escape(d.get("frequency") or "not specified")
        duration = html.escape(d.get("duration") or "not specified")
        st.markdown(
            f'<div class="field-card"><div class="field-title">{name}</div>{subtitle_html}'
            f'<div class="field-row"><b>Dosage:</b> {dosage}</div>'
            f'<div class="field-row"><b>Frequency:</b> {frequency}</div>'
            f'<div class="field-row"><b>Duration:</b> {duration}</div></div>',
            unsafe_allow_html=True,
        )


def render_discharge_card(normalized):
    # Single-line HTML per card - see render_prescription_cards for why.
    d = dataclasses.asdict(normalized)
    entities = d.get("diagnosis_entities") or []
    entity_chips = "".join(f'<span class="entity-chip">{html.escape(e)}</span>' for e in entities)
    chips_html = f'<div style="margin-top:0.5rem;">{entity_chips}</div>' if entity_chips else ""

    diagnosis = html.escape(d.get("canonical_diagnosis") or "not documented")
    procedures = html.escape(d.get("procedures") or "not documented")
    follow_up = html.escape(d.get("follow_up") or "not documented")

    st.markdown(
        f'<div class="field-card"><div class="field-title">Diagnosis</div>'
        f'<div class="field-row">{diagnosis}</div>{chips_html}</div>'
        f'<div class="field-card"><div class="field-title">Procedures</div>'
        f'<div class="field-row">{procedures}</div></div>'
        f'<div class="field-card"><div class="field-title">Follow-up</div>'
        f'<div class="field-row">{follow_up}</div></div>',
        unsafe_allow_html=True,
    )


def render_extracted_fields(doc_type: str, normalized):
    if doc_type == "lab":
        if not normalized:
            st.info("No structured fields were extracted from this document.")
        else:
            render_lab_cards(normalized)
    elif doc_type == "prescription":
        if not normalized:
            st.info("No structured fields were extracted from this document.")
        else:
            render_prescription_cards(normalized)
    else:
        render_discharge_card(normalized)


def render_flagged(doc_type: str, normalized):
    flagged = get_flagged_items(doc_type, normalized)
    if flagged is None:
        st.info("Prescriptions don't have a normal/abnormal flag in the structured data - see the Plain-Language Summary tab for medication-specific notes (e.g. unclear dosing instructions).")
        return
    if not flagged:
        st.success("No abnormal or flagged items found.")
        return

    # Single-line HTML per card - see render_prescription_cards for why
    # (a conditionally-empty line inside a multi-line f-string breaks
    # Streamlit's markdown/HTML rendering).
    if doc_type == "lab":
        for row in flagged:
            name = html.escape(row["canonical_test_name"] or row["test_name"])
            flag_label = html.escape(row.get("flag") or "")
            unit = html.escape(row.get("unit") or "")
            reference_range = html.escape(row.get("reference_range") or "not provided")
            st.markdown(
                f'<div class="flag-card"><div class="field-title">{name}'
                f'<span class="status-pill status-abnormal" style="float:right;">{flag_label}</span></div>'
                f'<div class="field-row"><b>{row["value"]} {unit}</b></div>'
                f'<div class="field-row">Reference range: {reference_range}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        for item in flagged:
            entity_chips = "".join(f'<span class="entity-chip">{html.escape(e)}</span>' for e in item.get("entities", []))
            chips_html = f'<div style="margin-top:0.5rem;">{entity_chips}</div>' if entity_chips else ""
            diagnosis = html.escape(item["diagnosis"])
            st.markdown(
                f'<div class="flag-card"><div class="field-title">{diagnosis}</div>{chips_html}</div>',
                unsafe_allow_html=True,
            )


def main():
    inject_custom_css()
    render_header()

    uploaded_file = st.file_uploader("Upload a report", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is None:
        st.info("Upload a file to get started. Sample files are available under data/samples/.")
        render_footer()
        return

    extension = os.path.splitext(uploaded_file.name)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        st.error(f"Unsupported file type: {extension}. Please upload a PDF, PNG, or JPG.")
        render_footer()
        return

    file_bytes = uploaded_file.getvalue()

    with st.spinner("\U0001F52C Analyzing report..."):
        result = run_pipeline(file_bytes, extension)

    if "error" in result:
        st.error(result["error"])
        if "raw_text" in result:
            with st.expander("Raw extracted text (extraction/normalization failed after this point)"):
                st.text(result["raw_text"])
        render_footer()
        return

    doc_label, doc_icon = DOC_TYPE_LABELS.get(result["doc_type"], (result["doc_type"].title(), "\U0001F4C4"))
    st.markdown(
        f'<div class="doc-type-badge">{doc_icon} Detected: {doc_label}</div>',
        unsafe_allow_html=True,
    )

    tab_raw, tab_fields, tab_summary, tab_flagged = st.tabs(
        ["\U0001F4C4 Raw Text", "\U0001F50D Extracted Fields", "\U0001F4DD Summary", "⚠️ Flagged Items"]
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

    render_footer()


if __name__ == "__main__":
    main()
