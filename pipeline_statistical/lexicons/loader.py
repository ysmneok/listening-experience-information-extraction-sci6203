import json
import unicodedata
from typing import Set


def normalize_term(term: str) -> str:
    term = unicodedata.normalize("NFKC", term)
    term = term.lower()
    return term.strip()


def load_lexicon(path) -> Set[str]:
    """
    Loads a lexicon JSON and returns a set of normalized terms.

    Supports:
    - flat list of strings
    - dict of lists (e.g. category -> [terms])
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    terms = set()

    if isinstance(data, list):
        terms.update(data)

    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                terms.update(value)
            elif isinstance(value, str):
                terms.add(value)

    else:
        raise ValueError("Unsupported lexicon format")

    return {normalize_term(t) for t in terms}
