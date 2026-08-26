"""
Heuristic document-type classifier.

Decides whether a medical report's raw text is a lab report, a
prescription, or a discharge summary by counting keyword hits per
category and picking the highest score. Simple on purpose - this is a
baseline to unblock the per-type extractors, not a trained model.
"""

import re

LAB_KEYWORDS = [
    "reference range", "test name", "specimen", "laboratory",
    "lab report", "hemoglobin", "wbc", "platelet", "cbc",
    "mg/dl", "/ul", "g/dl", "impression",
]

PRESCRIPTION_KEYWORDS = [
    "rx", "prescription", "dosage", "mg", "tablet", "capsule",
    "sig:", "refill", "take orally", "prescribing physician",
    "as needed", "daily",
]

DISCHARGE_KEYWORDS = [
    "discharge summary", "diagnosis", "follow-up", "follow up",
    "hospital course", "admitted", "discharged", "admission",
    "procedure", "procedures",
]

CATEGORY_KEYWORDS = {
    "lab": LAB_KEYWORDS,
    "prescription": PRESCRIPTION_KEYWORDS,
    "discharge": DISCHARGE_KEYWORDS,
}


def _score(text: str, keywords: list) -> int:
    return sum(len(re.findall(re.escape(kw), text)) for kw in keywords)


def classify_document(text: str) -> str:
    """
    Return "lab", "prescription", or "discharge" based on keyword hits.
    Defaults to "lab" if nothing matches (arbitrary fallback - there's
    no signal to base a smarter default on with keyword counting alone).
    """
    lowered = text.lower()
    scores = {category: _score(lowered, keywords) for category, keywords in CATEGORY_KEYWORDS.items()}
    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "lab"
    return best_category
