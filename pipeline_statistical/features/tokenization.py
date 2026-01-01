import re
from typing import List


def tokenize(text: str, ngrams=(1, 2)) -> List[str]:
    """
    Tokenize text into unigrams and n-grams (joined with underscores).

    Example:
    "very slow tempo" →
    ["very", "slow", "tempo", "very_slow", "slow_tempo"]
    """
    # Guard against None / empty input
    if not text:
        return []

    # Basic normalization
    text = text.lower()

    # Keep letters only
    tokens = re.findall(r"\b[a-z]+\b", text)

    results = tokens.copy()

    for n in ngrams:
        if n == 1:
            continue
        for i in range(len(tokens) - n + 1):
            results.append("_".join(tokens[i:i+n]))

    return results
