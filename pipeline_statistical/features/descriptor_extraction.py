from typing import List, Dict, Set, Tuple


def extract_descriptors(
    tokens: List[str],
    lexicon: Set[str]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Extract descriptor counts and binary presence from a tokenized document.

    Parameters
    ----------
    tokens : List[str]
        Tokenized document (already normalized).
    lexicon : Set[str]
        Controlled vocabulary (normalized surface forms).

    Returns
    -------
    counts : Dict[str, int]
        Descriptor -> frequency
    binary : Dict[str, int]
        Descriptor -> 0/1 presence
    """
    counts: Dict[str, int] = {}

    for tok in tokens:
        if tok in lexicon:
            counts[tok] = counts.get(tok, 0) + 1

    binary = {k: 1 for k in counts}

    return counts, binary

def extract_corpus_descriptors(
    corpus_tokens: Dict[str, List[str]],
    lexicon: Set[str]
) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    """
    Apply descriptor extraction to an entire corpus.

    Parameters
    ----------
    corpus_tokens : Dict[doc_id, List[str]]
        Tokenized corpus.
    lexicon : Set[str]

    Returns
    -------
    counts_list : List[Dict[str, int]]
        One dict per document.
    binary_list : List[Dict[str, int]]
        One dict per document.
    """
    counts_list = []
    binary_list = []

    for tokens in corpus_tokens.values():
        counts, binary = extract_descriptors(tokens, lexicon)
        counts_list.append(counts)
        binary_list.append(binary)

    return counts_list, binary_list
