"""
Splits raw report text into labeled sections based on heading lines.

A line counts as a heading if it's ONLY a short title - optionally ending
in a colon - with nothing else on it (e.g. "Diagnosis:" or "DISCHARGE
SUMMARY" on their own line). A line like "Patient: Jane Roe" is NOT a
heading because it has content after the colon. This keeps the detector
simple while avoiding the obvious false positive of treating every
"Label: value" line as a new section.
"""

import re

# A heading is either "Title:" (word(s) ending in a colon, nothing after)
# or an ALL-CAPS title line (e.g. "DISCHARGE SUMMARY"). Requiring one of
# these two explicit signals - rather than "any short capitalized line" -
# avoids misreading an ordinary sentence like "Echocardiogram performed
# on 2026-08-22" as a new section header.
HEADING_RE = re.compile(r'^(?:[A-Z][A-Za-z0-9 /\-]{1,40}:|[A-Z][A-Z0-9 /\-]{2,40})$')


def _normalize_key(heading: str) -> str:
    key = heading.strip().rstrip(":").strip().lower()
    key = re.sub(r"[\s/\-]+", "_", key)
    return key


def split_into_sections(text: str) -> dict:
    """
    Break text into {section_name: section_text} using heading lines as
    delimiters. Text before the first heading is stored under "preamble".
    """
    sections = {}
    current_key = "preamble"
    sections[current_key] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if HEADING_RE.match(line):
            current_key = _normalize_key(line)
            sections.setdefault(current_key, [])
            continue
        sections.setdefault(current_key, []).append(line)

    return {key: "\n".join(lines).strip() for key, lines in sections.items() if lines}


def get_section(sections: dict, candidate_keys: list) -> str:
    """
    Look up a section by trying several possible normalized key spellings
    (e.g. "follow_up" vs "followup"), since heading wording varies across
    reports. Returns "" if none of the candidates are present.
    """
    for key in candidate_keys:
        if key in sections:
            return sections[key]
    return ""


def get_section_by_keywords(sections: dict, keywords: list) -> str:
    """
    Fallback lookup for when no exact candidate heading matched: return
    the text of the first section (in document order) whose normalized
    key *contains* one of these keywords as a substring.

    Useful for concepts where real-world headings share a root word even
    though the exact phrasing varies too much to enumerate (e.g. any of
    "Procedures", "Procedure Notes", "Major Surgical or Invasive
    Procedure" all contain "procedure"). Not appropriate for concepts
    like "diagnosis", where real alternate headings (e.g. "Reason for
    Admission", "Chief Complaint") share no common substring at all -
    those still need explicit exact-match synonyms.
    """
    for key, text in sections.items():
        if any(kw in key for kw in keywords):
            return text
    return ""
