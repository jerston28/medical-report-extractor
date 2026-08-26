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
from extraction.section_splitter import split_into_sections, get_section


def extract_discharge_info(text: str) -> dict:
    """
    Extract discharge summary fields into a dict:
    {"diagnosis", "diagnosis_entities", "procedures", "follow_up"}
    """
    sections = split_into_sections(text)

    diagnosis_text = get_section(sections, ["diagnosis", "diagnoses"])
    procedures_text = get_section(sections, ["procedures", "procedure"])
    follow_up_text = get_section(sections, ["follow_up", "followup", "follow-up"])

    diagnosis_entities = [ent["text"] for ent in extract_entities(diagnosis_text)] if diagnosis_text else []

    return {
        "diagnosis": diagnosis_text or None,
        "diagnosis_entities": diagnosis_entities,
        "procedures": procedures_text or None,
        "follow_up": follow_up_text or None,
    }
