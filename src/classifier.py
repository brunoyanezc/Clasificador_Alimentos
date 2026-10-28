import numpy as np
import pandas as pd

from numpy.linalg import pinv


# =====================================================
# VALIDATION
# =====================================================

def validate_inputs(
    sample_vector,
    centroid_vector,
    uncertainty_vector
):
    """
    Validaciones básicas.
    """

    if len(sample_vector) == 0:
        raise ValueError(
            "No existen analitos para clasificar."
        )

    if len(sample_vector) != len(centroid_vector):
        raise ValueError(
            "sample_vector y centroid_vector "
            "tienen dimensiones distintas."
        )

    if len(sample_vector) != len(uncertainty_vector):
        raise ValueError(
            "uncertainty_vector tiene "
            "dimensiones incompatibles."
        )


# =====================================================
# DISTANCIA EUCLIDIANA
# =====================================================

def weighted_euclidean_distance(
    sample_vector,
    centroid_vector,
    uncertainty_vector
):
    """
    Distancia Euclidiana ponderada
    por incertidumbre analítica.
    """

    validate_inputs(
        sample_vector,
        centroid_vector,
        uncertainty_vector
    )

    uncertainty_vector = np.where(
        uncertainty_vector <= 0,
        1e-9,
        uncertainty_vector
    )

    z = (
        sample_vector
        - centroid_vector
    ) / uncertainty_vector

    return np.sqrt(
        np.sum(z ** 2)
    )


# =====================================================
# MATRIZ COVARIANZA
# =====================================================

def build_covariance_matrix(
    centroids_df,
    analytes,
    uncertainty_dict=None
):
    """
    Construye matriz de covarianza
    de los centroides.
    """

    X = (
        centroids_df[analytes]
        .astype(float)
        .values
    )

    cov = np.cov(
        X,
        rowvar=False
    )

    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    if uncertainty_dict is not None:

        noise = np.diag(
            [
                uncertainty_dict[a] ** 2
                for a in analytes
            ]
        )

        cov = cov + noise

    return cov


# =====================================================
# DISTANCIA MAHALANOBIS
# =====================================================

def mahalanobis_distance(
    sample_vector,
    centroid_vector,
    covariance_matrix
):
    """
    Distancia Mahalanobis.
    """

    diff = (
        sample_vector
        - centroid_vector
    )

    inverse_cov = pinv(
        covariance_matrix
    )

    distance_squared = (
        diff.T
        @ inverse_cov
        @ diff
    )

    distance_squared = max(
        float(distance_squared),
        0.0
    )

    return np.sqrt(
        distance_squared
    )


# =====================================================
# SELECCION DISTANCIA
# =====================================================

def calculate_distance(
    sample_vector,
    centroid_vector,
    uncertainty_vector,
    metric,
    covariance_matrix=None
):
    """
    Wrapper de métricas.
    """

    if metric == "euclidean":

        return weighted_euclidean_distance(
            sample_vector,
            centroid_vector,
            uncertainty_vector
        )

    if metric == "mahalanobis":

        if covariance_matrix is None:

            raise ValueError(
                "Covariance matrix requerida."
            )

        return mahalanobis_distance(
            sample_vector,
            centroid_vector,
            covariance_matrix
        )

    raise ValueError(
        f"Métrica no soportada: {metric}"
    )


# =====================================================
# SOFTMAX ESTABLE
# =====================================================

def softmax_from_distances(
    distances
):
    """
    Convierte distancias en probabilidades.
    """

    distances = np.array(
        distances,
        dtype=float
    )

    distances = (
        distances
        - distances.min()
    )

    scores = np.exp(
        -distances
    )

    denominator = scores.sum()

    if denominator <= 0:

        return np.zeros_like(
            scores
        )

    return (
        scores
        / denominator
    )


# =====================================================
# RANKING
# =====================================================

def build_ranking_dataframe(
    results
):
    """
    Construcción final dataframe.
    """

    ranking = pd.DataFrame(
        results
    )

    probabilities = (
        softmax_from_distances(
            ranking["Distance"]
        )
    )

    ranking[
        "Probability"
    ] = probabilities

    ranking = ranking.sort_values(
        "Probability",
        ascending=False
    )

    return ranking.reset_index(
        drop=True
    )


# =====================================================
# CLASIFICACION PRINCIPAL
# =====================================================

def classify_sample(
    sample_row,
    centroids_df,
    active_analytes,
    uncertainty_dict,
    metric="euclidean"
):
    """
    Clasificación principal.
    """

    sample_vector = (
        sample_row[
            active_analytes
        ]
        .astype(float)
        .values
    )

    uncertainty_vector = np.array(
        [
            uncertainty_dict[a]
            for a in active_analytes
        ]
    )

    use_mahalanobis = (
        metric == "mahalanobis"
    )

    covariance_matrix = None

    # -----------------------------------
    # Control estabilidad Mahalanobis
    # -----------------------------------

    if use_mahalanobis:

        if (
            len(centroids_df)
            <
            (
                len(active_analytes)
                + 1
            )
        ):

            # fallback automático

            metric = "euclidean"

        else:

            covariance_matrix = (
                build_covariance_matrix(
                    centroids_df,
                    active_analytes,
                    uncertainty_dict
                )
            )

    # -----------------------------------
    # Ranking
    # -----------------------------------

    results = []

    for _, row in centroids_df.iterrows():

        centroid_vector = (
            row[
                active_analytes
            ]
            .astype(float)
            .values
        )

        distance = calculate_distance(
            sample_vector=sample_vector,
            centroid_vector=centroid_vector,
            uncertainty_vector=uncertainty_vector,
            metric=metric,
            covariance_matrix=covariance_matrix
        )

        results.append(
            {
                "Product_Name":
                    row["Product_Name"],

                "Family":
                    row["Family"],


                "Distance":
                    distance,

                "Metric":
                    metric
            }
        )

    return build_ranking_dataframe(
        results
    )


# =====================================================
# UTILIDADES
# =====================================================

def get_best_match(
    ranking_df
):
    """
    Primera fila del ranking.
    """

    return ranking_df.iloc[0]


def get_top_matches(
    ranking_df,
    top_n=5
):
    """
    Top N candidatos.
    """

    return ranking_df.head(
        top_n
    ).copy()