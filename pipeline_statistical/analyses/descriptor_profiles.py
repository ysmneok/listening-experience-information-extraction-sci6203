# Analyses/descriptor_profiles.py

import pandas as pd


def build_descriptor_profiles(
    global_freq: pd.DataFrame,
    chi2_table: pd.DataFrame,
    ca_dim1: pd.DataFrame | None = None,
    dispersion: pd.Series | None = None,
    lexicon_name: str = "perceptual"
) -> list[dict]:
    """
    Build descriptor measurement profiles by consolidating
    existing analysis outputs.


    Parameters
    ----------
    global_freq : pd.DataFrame
        Output of global_frequency().
        Expected columns:
        - descriptorC descriptor
        - global_count
        - rank
        - percentile

    chi2_table : pd.DataFrame
        Output of chi2_descriptors_by_context().
        Expected columns:
        - descriptor
        - observed_<source>
        - expected_<source>
        - log_ratio_<source>

    ca_dim1 : pd.DataFrame, optional
        DataFrame with:
        - descriptor
        - contribution
        - cos2
        (from CA Dim 1)

    dispersion : pd.Series, optional
        Normalized dispersion per descriptor
        (index = descriptor)

    lexicon_name : str
        "perceptual" or "musico-technical"

    Returns
    -------
    list of dict
        One descriptor profile per descriptor
    """

    profiles = []

    # Ensure descriptor is index where needed
    global_freq = global_freq.set_index("descriptor")

    if ca_dim1 is not None:
        ca_dim1 = ca_dim1.set_index("descriptor")

    for _, row in chi2_table.iterrows():
        descriptor = row["descriptor"]

        profile = {
            "descriptor": descriptor,
            "lexicon": lexicon_name,
        }

        # ----------------------------
        # Global usage
        # ----------------------------
        if descriptor in global_freq.index:
            profile["global_usage"] = {
                "global_count": float(global_freq.loc[descriptor, "global_count"]),
                "global_rank": float(global_freq.loc[descriptor, "rank"]),
                "global_percentile": float(global_freq.loc[descriptor, "percentile"]),
            }
        else:
            profile["global_usage"] = None

        # ----------------------------
        # Usage by source
        # ----------------------------
        usage_by_source = {}

        for col in row.index:
            if col.startswith("observed_"):
                source = col.replace("observed_", "")
                usage_by_source[source] = {
                    "observed_count": float(row[col]),
                    "expected_count": float(row.get(f"expected_{source}", float("nan"))),
                    "log_ratio": float(row.get(f"log_ratio_{source}", float("nan"))),
                }

        profile["usage_by_source"] = usage_by_source

        # ----------------------------
        # Dispersion across genres
        # ----------------------------
        if dispersion is not None and descriptor in dispersion.index:
            profile["distribution_across_genres"] = {
                "normalized_dispersion": float(dispersion.loc[descriptor])
            }
        else:
            profile["distribution_across_genres"] = None

        # ----------------------------
        # Structural position (CA)
        # ----------------------------
        if ca_dim1 is not None and descriptor in ca_dim1.index:
            profile["structural_position"] = {
                "ca_dimension_1": {
                    "contribution": float(ca_dim1.loc[descriptor, "contribution"]),
                    "cos2": float(ca_dim1.loc[descriptor, "cos2"]),
                }
            }
        else:
            profile["structural_position"] = None

        profiles.append(profile)

    return profiles
