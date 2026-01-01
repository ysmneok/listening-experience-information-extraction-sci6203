# Analyses/diagnostics.py

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Global frequency diagnostics
# --------------------------------------------------

def global_frequency(
    df: pd.DataFrame,
    term_col: str = "descriptor",
    count_col: str = "count"
) -> pd.DataFrame:
    """
    Compute global frequency, rank, and percentile for each term.

    """
    gf = (
        df.groupby(term_col)[count_col]
        .sum()
        .sort_values(ascending=False)
        .to_frame("global_count")
    )
    gf["rank"] = gf["global_count"].rank(ascending=False, method="dense")
    gf["percentile"] = gf["global_count"].rank(pct=True)
    return gf


# --------------------------------------------------
# 2. Local relative frequency (normalization)
# --------------------------------------------------

def local_relative_frequency(
    df: pd.DataFrame,
    context_col: str,
    term_col: str,
    count_col: str
) -> pd.DataFrame:
    """
    Normalize term counts within each context.
    """
   
    totals = df.groupby(context_col)[count_col].sum()
    df = df.copy()
    df["relative_freq"] = df[count_col] / df[context_col].map(totals)
    return df


# --------------------------------------------------
# 3. Expected frequencies (χ² intuition)
# --------------------------------------------------

def expected_frequency(contingency: pd.DataFrame) -> pd.DataFrame:
    """
    Compute expected frequencies under independence.

    """
    row_totals = contingency.sum(axis=1)
    col_totals = contingency.sum(axis=0)
    grand_total = contingency.values.sum()

    expected = np.outer(row_totals, col_totals) / grand_total

    return pd.DataFrame(
        expected,
        index=contingency.index,
        columns=contingency.columns
    )


# --------------------------------------------------
# 4. Effect size: log ratio
# --------------------------------------------------

def log_ratio(
    observed: pd.DataFrame | pd.Series,
    expected: pd.DataFrame | pd.Series,
    eps: float = 1e-9
):
    """
    Compute log2(observed / expected).

    """
    return np.log2((observed + eps) / (expected + eps))


# --------------------------------------------------
# 5. Dispersion diagnostics
# --------------------------------------------------

def dispersion_entropy(counts: pd.Series) -> float:
    """
    Shannon entropy of a term across contexts.


    """
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-9)).sum())


def normalized_dispersion(counts: pd.Series) -> float:
    """
    Entropy normalized to [0, 1].

    """
    h = dispersion_entropy(counts)
    h_max = np.log2(len(counts))
    return float(h / h_max if h_max > 0 else 0.0)


# --------------------------------------------------
# 6. Local explanation helper (CA / χ² interpretation)
# --------------------------------------------------

def explain_descriptor(
    term: str,
    contingency: pd.DataFrame,
    expected: pd.DataFrame,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Returns top contexts by over-representation.
    """
    obs = contingency.loc[term]
    exp = expected.loc[term]
    lr = log_ratio(obs, exp)

    explanation = pd.DataFrame({
        "observed": obs,
        "expected": exp,
        "log_ratio": lr
    })

    return explanation.sort_values(
        "log_ratio",
        ascending=False
    ).head(top_n)


# --------------------------------------------------
# 7. Structural concentration diagnostics
# --------------------------------------------------

def pareto_mass(
    series: pd.Series,
    threshold: float = 0.8
) -> int:
    """
    Number of items accounting for a given share of total mass.
    """
    sorted_vals = series.sort_values(ascending=False)
    cumulative = sorted_vals.cumsum() / sorted_vals.sum()
    return int((cumulative <= threshold).sum())
