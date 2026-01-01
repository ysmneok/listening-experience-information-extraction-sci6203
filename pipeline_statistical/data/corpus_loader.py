import json
import logging
import re
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

EXCLUDED_GENRES = {
    "Blues",
    "Global",
    "Reggae",
    "New Age",
    "Latin Music"
}

QUOTE_PAIRS = [
    ('"', '"'),
    ('/', '/'),
    ('\\', '\\'),
    ('<', '>'),
]

MIN_QUOTED_CHARS = 30


def spans_in_quotes(text: str) -> List[Tuple[int, int]]:
    spans = []
    for open_q, close_q in QUOTE_PAIRS:
        pattern = re.compile(
            rf"{re.escape(open_q)}(.*?){re.escape(close_q)}",
            re.DOTALL
        )
        for m in pattern.finditer(text):
            if len(m.group(1)) >= MIN_QUOTED_CHARS:
                spans.append((m.start(), m.end()))
    return spans


def remove_quoted_spans(text: str) -> str:
    spans = spans_in_quotes(text)
    if not spans:
        return text

    cleaned = []
    last_idx = 0

    for start, end in sorted(spans):
        cleaned.append(text[last_idx:start])
        last_idx = end

    cleaned.append(text[last_idx:])
    return " ".join("".join(cleaned).split())


def load_corpus_metadata(path: str) -> List[Dict]:
    """
    Loads the review corpus JSON and returns a list of documents,
    excluding genres that do not support cross-source comparison
    and removing quoted material (e.g. song lyrics).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Corpus JSON must be a list of documents")

    filtered = []
    lyrics_removed_docs = 0

    for doc in data:
        if doc.get("genre") in EXCLUDED_GENRES:
            continue

        text = doc.get("review")
        if isinstance(text, str):
            cleaned_text = remove_quoted_spans(text)
            if cleaned_text != text:
                lyrics_removed_docs += 1
            doc = {**doc, "review": cleaned_text}

        filtered.append(doc)

    logger.info(
        f"Corpus filtering | "
        f"excluded_genres={sorted(EXCLUDED_GENRES)} | "
        f"removed_docs={len(data) - len(filtered)} | "
        f"kept_docs={len(filtered)} | "
        f"docs_with_lyrics_removed={lyrics_removed_docs}"
    )

    return filtered
