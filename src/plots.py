import matplotlib.pyplot as plt
import pandas as pd


def plot_pca(
    pca_df,
    sample_points=None
):
    """
    PCA de centroides.

    sample_points es opcional
    y corresponde a JOBs
    proyectados.
    """

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    families = (
        pca_df["Family"]
        .dropna()
        .unique()
    )

    cmap = plt.colormaps.get_cmap(
        "tab10"
    ).resampled(
        len(families)
    )

    family_colors = {
        family: cmap(i)
        for i, family in enumerate(
            families
        )
    }

    for family in families:

        subset = pca_df[
            pca_df["Family"] == family
        ]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=150,
            color=family_colors[family],
            label=family,
            alpha=0.8
        )

        for _, row in subset.iterrows():

            ax.annotate(
                row["Product_Name"],
                (
                    row["PC1"],
                    row["PC2"]
                ),
                fontsize=8
            )

    if (
        sample_points is not None
        and
        len(sample_points) > 0
    ):

        ax.scatter(
            sample_points["PC1"],
            sample_points["PC2"],
            marker="*",
            s=250,
            color="red",
            label="JOB"
        )

        for _, row in (
            sample_points.iterrows()
        ):

            ax.annotate(
                row["JOB"],
                (
                    row["PC1"],
                    row["PC2"]
                ),
                fontsize=9,
                fontweight="bold"
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.set_title(
        "PCA - Product Map"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    return fig


def plot_top_contributors(
    diagnostics_df,
    top_n=10
):
    """
    Gráfico de contribución
    por analito.
    """

    df = diagnostics_df.head(
        top_n
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        df["Analyte"],
        df["Contribution_%"]
    )

    ax.set_xlabel(
        "Contribution (%)"
    )

    ax.set_ylabel(
        "Analyte"
    )

    ax.set_title(
        "Main contributors"
    )

    ax.invert_yaxis()

    plt.tight_layout()

    return fig


def plot_probability_ranking(
    ranking_df,
    top_n=10
):
    """
    Ranking de Product_Name.
    """

    df = ranking_df.head(
        top_n
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        df["Product_Name"],
        df["Probability"] * 100
    )

    ax.set_xlabel(
        "Probability (%)"
    )

    ax.set_ylabel(
        "Product_Name"
    )

    ax.set_title(
        "Classification ranking"
    )

    ax.invert_yaxis()

    plt.tight_layout()

    return fig


def plot_distance_ranking(
    ranking_df,
    top_n=10
):
    """
    Ranking de distancias.
    """

    df = ranking_df.head(
        top_n
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        df["Product_Name"],
        df["Distance"]
    )

    ax.set_xlabel(
        "Distance"
    )

    ax.set_ylabel(
        "Product_Name"
    )

    ax.set_title(
        "Distance ranking"
    )

    ax.invert_yaxis()

    plt.tight_layout()

    return fig