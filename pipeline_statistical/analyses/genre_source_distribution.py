import logging
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Genre × Source distribution
# --------------------------------------------------

def genre_source_distribution(corpus):
    """
    Compute genre × source counts and proportions
    from an already-loaded (and filtered) corpus.

    Parameters
    ----------
    corpus : list of dict
        Each dict must contain at least 'genre' and 'source'

    Returns
    -------
    counts : pd.DataFrame
        Contingency table (genre × source)
    proportions : pd.DataFrame
        Column-normalized proportions (within each source)
    """

    rows = []

    for doc in corpus:
        genre = doc.get("genre")
        source = doc.get("source")

        if genre is None or source is None:
            continue

        rows.append({
            "genre": genre,
            "source": source
        })

    df = pd.DataFrame(rows)

    logger.info(f"Valid documents for genre × source distribution: {len(df)}")

    counts = pd.crosstab(df["genre"], df["source"])
    proportions = counts.div(counts.sum(axis=0), axis=1)

    return counts, proportions
