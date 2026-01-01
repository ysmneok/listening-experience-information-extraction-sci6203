import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


# --------------------------------------------------
# Stacked bar plot (proportions)
# --------------------------------------------------

def plot_genre_source_stacked_bars(proportions, figsize=(10, 6)):
    """
    Plot stacked bar chart of genre proportions by source.

    Parameters
    ----------
    proportions : pd.DataFrame
        genre × source proportions (columns sum to 1)
    """

    logger.info("Plotting genre × source stacked bar chart")

    ax = proportions.T.plot(
        kind="bar",
        stacked=True,
        figsize=figsize
    )

    ax.set_ylabel("Proportion of documents")
    ax.set_xlabel("Source")
    ax.set_title("Genre distribution by source")

    ax.legend(
        title="Genre",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Heatmap (counts or proportions)
# --------------------------------------------------

def plot_genre_source_heatmap(
    data,
    title="Genre × Source distribution",
    cmap="Blues",
    annot=True,
    fmt=".2f",
    figsize=(10, 8)
):
    """
    Plot heatmap for genre × source distribution.

    Parameters
    ----------
    data : pd.DataFrame
        genre × source table (counts or proportions)
    """

    logger.info("Plotting genre × source heatmap")

    plt.figure(figsize=figsize)

    sns.heatmap(
        data,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        cbar=True
    )

    plt.title(title)
    plt.ylabel("Genre")
    plt.xlabel("Source")

    plt.tight_layout()
    plt.show()
