import pandas as pd
import matplotlib.pyplot as plt


def plot_ca(
    row_coords: pd.DataFrame,
    col_coords: pd.DataFrame,
    dim_x: int = 1,
    dim_y: int = 2
):
    """
    Scatter plot of CA row and column coordinates.
    """
    dx = f"Dim_{dim_x}"
    dy = f"Dim_{dim_y}"

    plt.figure()

    plt.scatter(row_coords[dx], row_coords[dy], alpha=0.6)
    for label, x, y in zip(row_coords.index, row_coords[dx], row_coords[dy]):
        plt.text(x, y, label, fontsize=8, alpha=0.7)

    plt.scatter(col_coords[dx], col_coords[dy], marker="x", s=100)
    for label, x, y in zip(col_coords.index, col_coords[dx], col_coords[dy]):
        plt.text(x, y, label, fontsize=10, fontweight="bold")

    plt.axhline(0)
    plt.axvline(0)
    plt.xlabel(dx)
    plt.ylabel(dy)
    plt.title("Correspondence Analysis")
    plt.tight_layout()
    plt.show()
