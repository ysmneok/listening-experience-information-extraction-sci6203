import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def plot_top_chi2_residuals(
    chi2_df: pd.DataFrame,
    context: str,
    top_n: int = 20
):
    """
    Bar plot of top positive and negative standardized residuals
    for a given context.
    """

    col = f"residual_{context}"

    # --- Safety checks ---
    if chi2_df is None or chi2_df.empty:
        logger.warning("Chi-square DataFrame is empty — skipping plot")
        return

    if col not in chi2_df.columns:
        logger.warning(f"Column '{col}' not found — skipping plot")
        return

    df = chi2_df[["descriptor", col]].copy()

    # Drop invalid values that crash matplotlib
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        logger.warning("No finite chi-square residuals — skipping plot")
        return

    # Sort by residual
    df = df.sort_values(col)

    # Select top positive and negative
    top = pd.concat([
        df.head(top_n),
        df.tail(top_n)
    ])

    if top.empty:
        logger.warning("No residuals to plot after filtering — skipping plot")
        return

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    plt.barh(top["descriptor"], top[col])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Top χ² residuals — {context}")
    plt.xlabel("Standardized residual")
    plt.tight_layout()
    plt.show()
