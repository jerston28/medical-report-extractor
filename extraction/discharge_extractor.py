"""
Extraction of key fields from a discharge summary.

Discharge summaries are mostly free-flowing prose organized under
headings (Diagnosis, Procedures, Follow-up, ...), so regex alone can't
pull structured values the way it can for lab results. Instead this
locates the relevant sections by heading, then runs NER over the
diagnosis text to surface the specific clinical entities mentioned
(conditions), which is more useful downstream than the raw sentence.
"""

from extraction.ner_model import extract_entities
from extraction.section_splitter import split_into_sections, get_section, get_section_by_keywords

# Real discharge summaries word these headings differently from hospital
# to hospital. Diagnosis-equivalent headings are semantically diverse
# with no shared root word ("Reason for Admission" vs "Diagnosis" vs
# "Chief Complaint"), so they need an explicit synonym list. Procedures
# and follow-up headings, in practice, reliably share a root word
# ("procedure"/"surgical", "follow") even when the full phrasing
# varies, so those also fall back to substring keyword matching below
# when no exact match is found - this is deliberately broader than a
# fixed list so it generalizes to headings not seen yet.
DIAGNOSIS_KEYS = [
    "diagnosis", "diagnoses", "discharge_diagnosis", "discharge_diagnoses",
    "final_diagnosis", "primary_diagnosis", "reason_for_admission",
    "chief_complaint", "admitting_diagnosis", "impression",
]
PROCEDURES_KEYS = ["procedures", "procedure", "procedure_notes", "procedures_performed"]
PROCEDURES_KEYWORDS = ["procedure", "surgical", "surgery"]
FOLLOW_UP_KEYS = ["follow_up", "followup", "follow-up", "discharge_instructions", "follow_up_care", "follow_up_instructions"]
FOLLOW_UP_KEYWORDS = ["follow", "discharge_instruction", "discharge_plan"]


def extract_discharge_info(text: str) -> dict:
    """
    Extract discharge summary fields into a dict:
    {"diagnosis", "diagnosis_entities", "procedures", "follow_up"}
    """
    sections = split_into_sections(text)

    diagnosis_text = get_section(sections, DIAGNOSIS_KEYS)
    procedures_text = get_section(sections, PROCEDURES_KEYS) or get_section_by_keywords(sections, PROCEDURES_KEYWORDS)
    follow_up_text = get_section(sections, FOLLOW_UP_KEYS) or get_section_by_keywords(sections, FOLLOW_UP_KEYWORDS)

    diagnosis_entities = [ent["text"] for ent in extract_entities(diagnosis_text)] if diagnosis_text else []

    return {
        "diagnosis": diagnosis_text or None,
        "diagnosis_entities": diagnosis_entities,
        "procedures": procedures_text or None,
        "follow_up": follow_up_text or None,
    }
