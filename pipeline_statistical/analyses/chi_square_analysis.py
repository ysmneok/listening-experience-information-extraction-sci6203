# Analyses/chi_square_analysis.py

import numpy as np
import pandas as pd
import logging
from scipy.sparse import csr_matrix
from scipy.stats import chi2_contingency
from typing import List

from pipeline_statistical.analyses.diagnostics import log_ratio

logger = logging.getLogger(__name__)


def chi2_descriptors_by_context(
    X_binary: csr_matrix,
    feature_names: List[str],
    contexts: List[str],
    min_doc_freq: int = 5
) -> pd.DataFrame:
    """
    Chi-square analysis of descriptor presence by context
    with interpretable diagnostics.

    Returns a table you can reason about.
    """
    contexts = np.asarray(contexts)
    unique_contexts = np.unique(contexts)

    logger.info(f"Running chi-square on {len(feature_names)} descriptors")
    logger.info(f"Contexts: {list(unique_contexts)}")

    rows = []

    for j, descriptor in enumerate(feature_names):
        col = X_binary[:, j].toarray().ravel()
        doc_freq = col.sum()

        if doc_freq < min_doc_freq:
            continue

        # Build contingency table: rows=context, cols=[present, absent]
        contingency = []
        observed_present = {}

        for ctx in unique_contexts:
            mask = contexts == ctx
            present = col[mask].sum()
            absent = mask.sum() - present
            contingency.append([present, absent])
            observed_present[ctx] = present

        contingency = np.array(contingency)

        try:
            chi2, p, dof, expected = chi2_contingency(contingency)
        except ValueError:
            continue

        # Expected present counts are in expected[:, 0]
        expected_present = {
            ctx: expected[i, 0]
            for i, ctx in enumerate(unique_contexts)
        }

        # Effect size (log-ratio on present counts)
        obs_series = pd.Series(observed_present)
        exp_series = pd.Series(expected_present)
        lr = log_ratio(obs_series, exp_series)

        # Standardized residuals (present only)
        residuals = (obs_series - exp_series) / np.sqrt(exp_series)

        row = {
            "descriptor": descriptor,
            "chi2": chi2,
            "p_value": p,
            "dof": dof,
        }

        for ctx in unique_contexts:
            row[f"observed_{ctx}"] = obs_series[ctx]
            row[f"expected_{ctx}"] = exp_series[ctx]
            row[f"log_ratio_{ctx}"] = lr[ctx]
            row[f"residual_{ctx}"] = residuals[ctx]

        rows.append(row)

    df = pd.DataFrame(rows)

    logger.info(f"Descriptors tested: {len(df)}")

    return df.sort_values("chi2", ascending=False)
