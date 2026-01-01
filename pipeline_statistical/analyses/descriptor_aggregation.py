import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List


def aggregate_descriptors_global(
    X_counts: csr_matrix,
    X_binary: csr_matrix,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Compute global descriptor statistics.

    Returns both absolute counts and normalized measures.

    Columns:
    - descriptor
    - total_count           (total occurrences in corpus)
    - doc_count             (number of documents containing descriptor)
    - document_frequency    (doc_count / n_docs)
    - mean_frequency        (total_count / n_docs)
    """
    n_docs = X_counts.shape[0]

    total_count = np.asarray(X_counts.sum(axis=0)).ravel()
    doc_count = np.asarray(X_binary.sum(axis=0)).ravel()

    document_frequency = doc_count / n_docs
    mean_frequency = total_count / n_docs

    return pd.DataFrame({
        "descriptor": feature_names,
        "total_count": total_count,
        "doc_count": doc_count,
        "document_frequency": document_frequency,
        "mean_frequency": mean_frequency
    })


def aggregate_descriptors_by_context(
    X_counts: csr_matrix,
    X_binary: csr_matrix,
    feature_names: List[str],
    contexts: List[str]
) -> pd.DataFrame:
    """
    Aggregate descriptor statistics by a categorical context
    (e.g. source, genre).

    Columns:
    - descriptor
    - context
    - total_count
    - doc_count
    - document_frequency
    - mean_frequency
    """
    contexts = np.asarray(contexts)
    results = []

    for ctx in np.unique(contexts):
        mask = contexts == ctx
        n_docs = mask.sum()

        if n_docs == 0:
            continue

        Xc = X_counts[mask]
        Xb = X_binary[mask]

        total_count = np.asarray(Xc.sum(axis=0)).ravel()
        doc_count = np.asarray(Xb.sum(axis=0)).ravel()

        document_frequency = doc_count / n_docs
        mean_frequency = total_count / n_docs

        df = pd.DataFrame({
            "descriptor": feature_names,
            "context": ctx,
            "total_count": total_count,
            "doc_count": doc_count,
            "document_frequency": document_frequency,
            "mean_frequency": mean_frequency
        })

        results.append(df)

    return pd.concat(results, ignore_index=True)
