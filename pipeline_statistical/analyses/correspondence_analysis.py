# Analyses/correspondence_analysis.py

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.decomposition import TruncatedSVD


def correspondence_analysis(
    contingency_table: pd.DataFrame,
    n_components: int = 2
) -> Dict[str, object]:
    """
    Performs Correspondence Analysis (CA) with full diagnostics.

    Returns a dict containing:
    - row_coords
    - col_coords
    - singular_values
    - row_masses
    - col_masses
    - expected
    - residuals
    - row_contrib
    - col_contrib
    - row_cos2
    """

    # --------------------------------------------------
    # 1. Clean contingency table
    # --------------------------------------------------
    contingency = contingency_table.loc[
        contingency_table.sum(axis=1) > 0,
        contingency_table.sum(axis=0) > 0
    ]

    if contingency.empty:
        raise ValueError("Contingency table is empty after filtering.")

    N = contingency.values.astype(float)
    grand_total = N.sum()
    if grand_total == 0:
        raise ValueError("Contingency table has zero mass.")

    # --------------------------------------------------
    # 2. Correspondence matrix and marginals
    # --------------------------------------------------
    P = N / grand_total
    r = P.sum(axis=1)  # row masses
    c = P.sum(axis=0)  # column masses

    row_masses = pd.Series(r, index=contingency.index, name="row_mass")
    col_masses = pd.Series(c, index=contingency.columns, name="col_mass")

    # Expected frequencies under independence
    expected = np.outer(r, c)
    expected_df = pd.DataFrame(
        expected * grand_total,
        index=contingency.index,
        columns=contingency.columns
    )

    # Standardized residuals
    residuals = (contingency - expected_df) / np.sqrt(expected_df)

    # --------------------------------------------------
    # 3. Dimensionality guard
    # --------------------------------------------------
    max_components = min(len(r) - 1, len(c) - 1)
    if max_components < 1:
        raise ValueError("Not enough dimensions for CA.")

    n_components = min(n_components, max_components)

    # --------------------------------------------------
    # 4. CA core (χ² geometry)
    # --------------------------------------------------
    Dr_inv_sqrt = np.diag(1.0 / np.sqrt(r))
    Dc_inv_sqrt = np.diag(1.0 / np.sqrt(c))

    S = Dr_inv_sqrt @ (P - np.outer(r, c)) @ Dc_inv_sqrt

    svd = TruncatedSVD(n_components=n_components)
    U = svd.fit_transform(S)
    V = svd.components_.T
    singular_values = svd.singular_values_

    eigvals = singular_values ** 2

    # --------------------------------------------------
    # 5. Coordinates
    # --------------------------------------------------
    row_coords = Dr_inv_sqrt @ U
    col_coords = Dc_inv_sqrt @ V @ np.diag(singular_values)

    row_coords = pd.DataFrame(
        row_coords,
        index=contingency.index,
        columns=[f"Dim_{i+1}" for i in range(n_components)]
    )

    col_coords = pd.DataFrame(
        col_coords,
        index=contingency.columns,
        columns=[f"Dim_{i+1}" for i in range(n_components)]
    )

    # --------------------------------------------------
    # 6. Contributions
    # --------------------------------------------------
    row_contrib = pd.DataFrame(
        {
            f"Dim_{k+1}": (row_masses.values * row_coords.iloc[:, k] ** 2) / eigvals[k]
            for k in range(n_components)
        },
        index=row_coords.index
    )

    col_contrib = pd.DataFrame(
        {
            f"Dim_{k+1}": (col_masses.values * col_coords.iloc[:, k] ** 2) / eigvals[k]
            for k in range(n_components)
        },
        index=col_coords.index
    )

    # --------------------------------------------------
    # 7. Cos² (quality of representation)
    # --------------------------------------------------
    row_cos2 = row_coords ** 2
    row_cos2 = row_cos2.div(row_cos2.sum(axis=1), axis=0)

    col_cos2 = col_coords ** 2
    col_cos2 = col_cos2.div(col_cos2.sum(axis=1), axis=0)

    # --------------------------------------------------
    # 8. Return structured result
    # --------------------------------------------------
    return {
        "row_coords": row_coords,
        "col_coords": col_coords,
        "singular_values": singular_values,
        "eigenvalues": eigvals,
        "row_masses": row_masses,
        "col_masses": col_masses,
        "expected": expected_df,
        "residuals": residuals,
        "row_contrib": row_contrib,
        "col_contrib": col_contrib,
        "row_cos2": row_cos2,
        "col_cos2": col_cos2,
    }
