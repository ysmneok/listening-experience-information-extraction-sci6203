# Analyses/descriptor_reports.py

import pandas as pd

from pipeline_statistical.analyses.diagnostics import (
    global_frequency,
    normalized_dispersion,
    expected_frequency,
    log_ratio,
)


def build_descriptor_report(
    term: str,
    contingency: pd.DataFrame,
    global_counts: pd.DataFrame
) -> pd.DataFrame:
    """
    Full diagnostic report for a single descriptor.


    """

    if term not in contingency.index:
        raise ValueError(f"Descriptor '{term}' not found in contingency table.")

    # Global importance
    global_info = global_counts.loc[term]

    # Observed / expected
    observed = contingency.loc[term]
    expected = expected_frequency(contingency).loc[term]
    lr = log_ratio(observed, expected)

    # Dispersion
    dispersion = normalized_dispersion(observed)

    report = pd.DataFrame({
        "observed": observed,
        "expected": expected,
        "log_ratio": lr,
    })

    report["descriptor"] = term
    report["global_count"] = global_info["global_count"]
    report["global_rank"] = global_info["rank"]
    report["global_percentile"] = global_info["percentile"]
    report["dispersion"] = dispersion

    return report.sort_values("log_ratio", ascending=False)
