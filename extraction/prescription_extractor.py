"""
Extraction of prescription line items (drug, dosage, frequency, duration).

Approach: regex finds the structured, predictable parts of an Rx line -
dosage amounts, frequency phrasing, duration - since those follow a
handful of common patterns. The drug name itself is free text with no
fixed pattern, so instead of guessing with more regex we lean on the
scispaCy NER model to confirm which words in the line are a recognized
clinical entity.
"""

import re

from extraction.ner_model import extract_entities
from extraction.section_splitter import split_into_sections, get_section

DOSAGE_RE = re.compile(r"\d+\.?\d*\s*(?:mg|mcg|g|ml|iu|units?)", re.IGNORECASE)

FREQUENCY_RE = re.compile(
    r"\d+x\s*daily|once daily|twice daily|three times daily|four times daily|"
    r"every\s+\d+\s*hours?|as needed|\bbid\b|\btid\b|\bqid\b",
    re.IGNORECASE,
)

DURATION_RE = re.compile(r"for\s+\d+\s*(?:days|day|weeks|week)", re.IGNORECASE)

LEADING_NUMBERING_RE = re.compile(r"^\s*\d+[.)]\s*")


def _extract_drug_name(line: str, dosage_match) -> str:
    """
    The drug name is whatever text sits before the dosage amount, once
    list numbering ("1.") is stripped off.
    """
    before_dosage = line[:dosage_match.start()]
    before_dosage = LEADING_NUMBERING_RE.sub("", before_dosage)
    return before_dosage.strip(" -:")


def extract_prescription_info(text: str) -> list:
    """
    Extract prescription line items into a list of dicts:
    {"drug", "dosage", "frequency", "duration", "ner_confirmed"}
    """
    sections = split_into_sections(text)
    rx_text = get_section(sections, ["rx", "prescription", "medications"]) or text

    entities = extract_entities(rx_text)
    entity_texts_lower = {ent["text"].lower() for ent in entities}

    results = []
    for line in rx_text.splitlines():
        line = line.strip()
        if not line:
            continue

        dosage_match = DOSAGE_RE.search(line)
        if not dosage_match:
            continue

        drug = _extract_drug_name(line, dosage_match)
        if not drug:
            continue

        frequency_match = FREQUENCY_RE.search(line)
        duration_match = DURATION_RE.search(line)

        ner_confirmed = any(
            drug.lower() in entity_text or entity_text in drug.lower()
            for entity_text in entity_texts_lower
        )

        results.append({
            "drug": drug,
            "dosage": dosage_match.group().strip(),
            "frequency": frequency_match.group().strip() if frequency_match else None,
            "duration": duration_match.group().strip() if duration_match else None,
            "ner_confirmed": ner_confirmed,
        })

    return results
