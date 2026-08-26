"""
Shared scispaCy NER model loader.

en_core_sci_sm is a biomedical NER model - it recognizes mentions of
clinical concepts (drugs, diseases, procedures, anatomy, etc.) in text,
but tags them all with a generic "ENTITY" label rather than fine-grained
types like DRUG or DISEASE. That's still useful: it tells the
per-type extractors *which spans of text are clinically meaningful*, which
regex alone can't determine.

The model is loaded once and cached at module level, since loading it
takes a noticeable amount of time and every extractor needs it.
"""

import spacy

_nlp = None


def load_ner_model():
    """
    Load (once) and return the scispaCy en_core_sci_sm pipeline.
    """
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_sci_sm")
    return _nlp


def extract_entities(text: str) -> list:
    """
    Run NER over text and return a list of dicts:
    {"text": entity string, "label": entity type, "start": char offset, "end": char offset}
    """
    nlp = load_ner_model()
    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]
