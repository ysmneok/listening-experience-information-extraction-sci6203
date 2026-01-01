import json
import logging
import time
import os
import sys
from pathlib import Path
from collections import Counter

# Add project root to path for config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PERCEPTUAL_LEXICON_PATH,
    MUSICO_TECH_LEXICON_PATH,
    CORPUS_PATH
)

# --- Loaders ---
from pipeline_statistical.data.corpus_loader import load_corpus_metadata
from pipeline_statistical.lexicons.loader import load_lexicon

# --- Feature extraction ---
from pipeline_statistical.features.descriptor_extraction import extract_corpus_descriptors
from pipeline_statistical.features.tokenization import tokenize

# --- Vectorization ---
from pipeline_statistical.features.vectorization_matrices import build_attribute_matrices

# --- Analyses ---
from pipeline_statistical.analyses.chi_square_analysis import chi2_descriptors_by_context
from pipeline_statistical.analyses.correspondence_analysis import correspondence_analysis
from pipeline_statistical.analyses.descriptor_aggregation import (
    aggregate_descriptors_global,
    aggregate_descriptors_by_context
)
from pipeline_statistical.analyses.genre_source_distribution import genre_source_distribution
from pipeline_statistical.analyses.diagnostics import (
    global_frequency,
    local_relative_frequency,
    pareto_mass,
    normalized_dispersion
)
from pipeline_statistical.analyses.descriptor_reports import build_descriptor_report
from pipeline_statistical.analyses.descriptor_profiles import build_descriptor_profiles

# --- Reporting ---
from pipeline_statistical.reporting.chi_square_plots import plot_top_chi2_residuals
from pipeline_statistical.reporting.correspondence_plots import plot_ca
from pipeline_statistical.reporting.genre_source_plots import (
    plot_genre_source_stacked_bars,
    plot_genre_source_heatmap
)


# --------------------
# Logging setup
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def _safe_valid_genres(genres, min_docs=50):
    genre_counts = Counter(genres)
    return {g for g, c in genre_counts.items() if c >= min_docs}


def _contingency_from_by_context(df, value_col="total_count"):
    if value_col not in df.columns:
        raise ValueError(
            f"Expected column '{value_col}' in aggregation output, "
            f"but got columns={list(df.columns)}"
        )

    return (
        df.groupby(["descriptor", "context"])[value_col]
        .sum()
        .unstack(fill_value=0)
    )


def main():
    logger.info("PIPELINE START")
    t0 = time.perf_counter()

    # --------------------------------------------------
    # 1. Load corpus and lexicons
    # --------------------------------------------------
    corpus = load_corpus_metadata(CORPUS_PATH)
    perceptual_lexicon = load_lexicon(PERCEPTUAL_LEXICON_PATH)
    musico_lexicon = load_lexicon(MUSICO_TECH_LEXICON_PATH)

    logger.info(f"Corpus loaded | n_docs={len(corpus)}")
    logger.info(f"Perceptual lexicon | n_terms={len(perceptual_lexicon)}")
    logger.info(f"Musico-tech lexicon | n_terms={len(musico_lexicon)}")

    # --------------------------------------------------
    # 1b. Genre × source distribution
    # --------------------------------------------------
    genre_source_counts, genre_source_props = genre_source_distribution(corpus)

    plot_genre_source_stacked_bars(genre_source_props)
    plot_genre_source_heatmap(
        genre_source_counts,
        title="Genre × Source document counts",
        fmt="d"
    )

    # --------------------------------------------------
    # 2. Tokenization
    # --------------------------------------------------
    corpus_tokens = {i: tokenize(doc["review"]) for i, doc in enumerate(corpus)}
    contexts = [doc.get("source", "UNKNOWN") for doc in corpus]
    genres = [doc.get("genre", "UNKNOWN") for doc in corpus]

    logger.info(f"Context distribution | {dict(Counter(contexts))}")
    logger.info(f"Genre distribution | {dict(Counter(genres))}")

    # --------------------------------------------------
    # 3. Descriptor extraction
    # --------------------------------------------------
    p_counts, p_binary = extract_corpus_descriptors(corpus_tokens, perceptual_lexicon)
    m_counts, m_binary = extract_corpus_descriptors(corpus_tokens, musico_lexicon)

    # --------------------------------------------------
    # 4. Vectorization
    # --------------------------------------------------
    Xp_counts, Xp_binary, p_features = build_attribute_matrices(p_counts, p_binary)
    Xm_counts, Xm_binary, m_features = build_attribute_matrices(m_counts, m_binary)

    # ==================================================
    # PERCEPTUAL
    # ==================================================

    p_global_long = aggregate_descriptors_global(Xp_counts, Xp_binary, p_features)
    p_global_freq = global_frequency(p_global_long, "descriptor", "total_count")

    chi2_p_source = chi2_descriptors_by_context(
        X_binary=Xp_binary,
        feature_names=p_features,
        contexts=contexts
    )
    plot_top_chi2_residuals(chi2_p_source, context=contexts[0])

    p_by_source = aggregate_descriptors_by_context(
        Xp_counts, Xp_binary, p_features, contexts
    )
    contingency_p_source = _contingency_from_by_context(p_by_source)
    dispersion_p_source = contingency_p_source.apply(normalized_dispersion, axis=1)

    top_p = chi2_p_source.iloc[0]["descriptor"]
    descriptor_report_p = build_descriptor_report(
        term=top_p,
        contingency=contingency_p_source,
        global_counts=p_global_freq
    )

    ca_p_source = correspondence_analysis(contingency_p_source)
    if ca_p_source["row_coords"].shape[1] >= 2:
        plot_ca(ca_p_source["row_coords"], ca_p_source["col_coords"])
    else:
        logger.info("Perceptual × source CA is 1D; skipping plot.")

    ca_p_source_dim1 = (
        ca_p_source["row_contrib"]["Dim_1"]
        .to_frame("contribution")
        .join(ca_p_source["row_cos2"]["Dim_1"].to_frame("cos2"))
        .reset_index()
    )

    p_by_genre = aggregate_descriptors_by_context(
        Xp_counts, Xp_binary, p_features, genres
    )
    p_by_genre = p_by_genre[p_by_genre["context"].isin(_safe_valid_genres(genres))]

    contingency_p_genre = (
        p_by_genre
        .pivot(index="descriptor", columns="context", values="mean_frequency")
        .fillna(0)
    )

    ca_p_genre = correspondence_analysis(contingency_p_genre)
    if ca_p_genre["row_coords"].shape[1] >= 2:
        plot_ca(ca_p_genre["row_coords"], ca_p_genre["col_coords"])
    else:
        logger.info("Perceptual × genre CA is 1D; skipping plot.")

    # ==================================================
    # MUSICO-TECH
    # ==================================================

    m_global_long = aggregate_descriptors_global(Xm_counts, Xm_binary, m_features)
    m_global_freq = global_frequency(m_global_long, "descriptor", "total_count")

    chi2_m_source = chi2_descriptors_by_context(
        X_binary=Xm_binary,
        feature_names=m_features,
        contexts=contexts
    )

    m_by_source = aggregate_descriptors_by_context(
        Xm_counts, Xm_binary, m_features, contexts
    )
    contingency_m_source = _contingency_from_by_context(m_by_source)
    dispersion_m_source = contingency_m_source.apply(normalized_dispersion, axis=1)

    ca_m_source = correspondence_analysis(contingency_m_source)
    if ca_m_source["row_coords"].shape[1] >= 2:
        plot_ca(ca_m_source["row_coords"], ca_m_source["col_coords"])
    else:
        logger.info("Musico-tech × source CA is 1D; skipping plot.")

    ca_m_source_dim1 = (
        ca_m_source["row_contrib"]["Dim_1"]
        .to_frame("contribution")
        .join(ca_m_source["row_cos2"]["Dim_1"].to_frame("cos2"))
        .reset_index()
    )

    m_by_genre = aggregate_descriptors_by_context(
        Xm_counts, Xm_binary, m_features, genres
    )
    m_by_genre = m_by_genre[m_by_genre["context"].isin(_safe_valid_genres(genres))]

    contingency_m_genre = (
        m_by_genre
        .pivot(index="descriptor", columns="context", values="mean_frequency")
        .fillna(0)
    )

    ca_m_genre = correspondence_analysis(contingency_m_genre)
    if ca_m_genre["row_coords"].shape[1] >= 2:
        plot_ca(ca_m_genre["row_coords"], ca_m_genre["col_coords"])
    else:
        logger.info("Musico-tech × genre CA is 1D; skipping plot.")

    # --------------------------------------------------
    # EXPORT RESULTS
    # --------------------------------------------------
    logger.info("Exporting outputs")
    # Outputs go to pipeline_statistical/Outputs/
    OUTPUTS_DIR = Path(__file__).resolve().parent / "Outputs"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    p_global_freq.reset_index().to_json(
        str(OUTPUTS_DIR / "global_perceptual_descriptors.json"),
        orient="records",
        indent=2
    )

    m_global_freq.reset_index().to_json(
        str(OUTPUTS_DIR / "global_musico_tech_descriptors.json"),
        orient="records",
        indent=2
    )

    chi2_p_source.to_json(
        str(OUTPUTS_DIR / "chi2_perceptual_source.json"),
        orient="records",
        indent=2
    )

    chi2_m_source.to_json(
        str(OUTPUTS_DIR / "chi2_musico_tech_source.json"),
        orient="records",
        indent=2
    )

    ca_p_source_dim1.to_json(
        str(OUTPUTS_DIR / "ca_dim1_perceptual_source.json"),
        orient="records",
        indent=2
    )

    ca_m_source_dim1.to_json(
        str(OUTPUTS_DIR / "ca_dim1_musico_tech_source.json"),
        orient="records",
        indent=2
    )

    ca_p_genre["row_contrib"]["Dim_1"] \
        .to_frame("contribution") \
        .join(ca_p_genre["row_cos2"]["Dim_1"].to_frame("cos2")) \
        .reset_index() \
        .to_json(
            str(OUTPUTS_DIR / "ca_dim1_perceptual_genre.json"),
            orient="records",
            indent=2
        )

    ca_m_genre["row_contrib"]["Dim_1"] \
        .to_frame("contribution") \
        .join(ca_m_genre["row_cos2"]["Dim_1"].to_frame("cos2")) \
        .reset_index() \
        .to_json(
            str(OUTPUTS_DIR / "ca_dim1_musico_tech_genre.json"),
            orient="records",
            indent=2
        )

    descriptor_report_p.reset_index().to_json(
        str(OUTPUTS_DIR / f"descriptor_report_{top_p}.json"),
        orient="records",
        indent=2
    )

    logger.info(f"All outputs written to {OUTPUTS_DIR}")
    logger.info(f"PIPELINE END | total_time={time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
