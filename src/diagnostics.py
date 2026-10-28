import numpy as np
import pandas as pd


# =====================================================
# CONTRIBUCION POR ANALITO
# =====================================================

def analyte_contributions(
    sample_row,
    centroid_row,
    active_analytes,
    uncertainty_dict
):
    """
    Calcula la contribución individual
    de cada analito a la distancia total.
    """

    sample_vector = (
        sample_row[
            active_analytes
        ]
        .astype(float)
        .values
    )

    centroid_vector = (
        centroid_row[
            active_analytes
        ]
        .astype(float)
        .values
    )

    uncertainty_vector = np.array(
        [
            uncertainty_dict[a]
            for a in active_analytes
        ],
        dtype=float
    )

    uncertainty_vector = np.where(
        uncertainty_vector <= 0,
        1e-9,
        uncertainty_vector
    )

    difference = (
        sample_vector
        - centroid_vector
    )

    z_scores = (
        difference
        / uncertainty_vector
    )

    squared_contribution = (
        z_scores ** 2
    )

    total = squared_contribution.sum()

    if total > 0:

        contribution_pct = (
            squared_contribution
            / total
        ) * 100

    else:

        contribution_pct = np.zeros(
            len(active_analytes)
        )

    result = pd.DataFrame(
        {
            "Analyte":
                active_analytes,

            "Sample_Value":
                sample_vector,

            "Target_Value":
                centroid_vector,

            "Difference":
                difference,

            "Uncertainty":
                uncertainty_vector,

            "Zscore":
                z_scores,

            "Contribution_%":
                contribution_pct
        }
    )

    result = result.sort_values(
        "Contribution_%",
        ascending=False
    )

    return result.reset_index(
        drop=True
    )


# =====================================================
# COBERTURA
# =====================================================

def coverage_ratio(
    active_analytes,
    selected_analytes
):
    """
    Fracción de analitos efectivamente
    utilizados respecto a los
    seleccionados por el usuario.
    """

    denominator = len(
        selected_analytes
    )

    if denominator == 0:

        return 0.0

    return (
        len(active_analytes)
        / denominator
    )


def coverage_quality(
    coverage
):
    """
    Interpretación cualitativa.
    """

    if coverage >= 0.80:

        return "Alta"

    if coverage >= 0.50:

        return "Media"

    return "Baja"


# =====================================================
# ANALITOS EXCLUIDOS
# =====================================================

def get_excluded_analytes(
    selected_analytes,
    active_analytes
):
    """
    Analitos seleccionados por el usuario
    pero no utilizados finalmente.
    """

    return sorted(
        list(
            set(selected_analytes)
            -
            set(active_analytes)
        )
    )


# =====================================================
# METADATA
# =====================================================

def diagnostics_metadata(
    selected_analytes,
    active_analytes
):
    """
    Resumen para la UI.
    """

    coverage = coverage_ratio(
        active_analytes,
        selected_analytes
    )

    excluded = get_excluded_analytes(
        selected_analytes,
        active_analytes
    )

    return {
        "Selected":
            len(selected_analytes),

        "Used":
            len(active_analytes),

        "Excluded":
            len(excluded),

        "Excluded_Analytes":
            excluded,

        "Coverage":
            coverage,

        "Coverage_Quality":
            coverage_quality(
                coverage
            )
    }


# =====================================================
# TOP CONTRIBUYENTES
# =====================================================

def summarize_top_analytes(
    diagnostics_df,
    top_n=3
):
    """
    Texto ejecutivo para reportes.
    """

    top = diagnostics_df.head(
        top_n
    )

    messages = []

    for _, row in top.iterrows():

        messages.append(
            (
                f"{row['Analyte']} "
                f"(Z={row['Zscore']:.2f}, "
                f"{row['Contribution_%']:.1f}%)"
            )
        )

    return "; ".join(
        messages
    )


# =====================================================
# SCORE DE CONFIANZA
# =====================================================

def confidence_from_probability(
    probability
):
    """
    Traducción a lenguaje natural.
    """

    if probability >= 0.90:

        return "Muy Alta"

    if probability >= 0.75:

        return "Alta"

    if probability >= 0.50:

        return "Media"

    return "Baja"


# =====================================================
# REPORTE EJECUTIVO
# =====================================================

def generate_classification_summary(
    job,
    product_name,
    family,
    probability,
    distance,
    coverage,
    diagnostics_df
):
    """
    Construye resumen ejecutivo
    para una muestra.
    """

    return {
        "JOB":
            job,

        "Product_Name":
            product_name,

        "Family":
            family,

        "Probability":
            float(probability),

        "Confidence":
            confidence_from_probability(
                float(probability)
            ),

        "Distance":
            float(distance),

        "Coverage":
            float(coverage),

        "Coverage_Quality":
            coverage_quality(
                coverage
            ),

        "Main_Contributors":
            summarize_top_analytes(
                diagnostics_df
            )
    }


# =====================================================
# REPORTE DESDE RANKING
# =====================================================

def build_job_report(
    job,
    ranking_df,
    diagnostics_df,
    coverage
):
    """
    Construcción automática
    desde el ranking obtenido
    en classifier.py.
    """

    winner = ranking_df.iloc[0]

    return generate_classification_summary(
        job=job,
        product_name=winner[
            "Product_Name"
        ],
        family=winner[
            "Family"
        ],
        probability=winner[
            "Probability"
        ],
        distance=winner[
            "Distance"
        ],
        coverage=coverage,
        diagnostics_df=diagnostics_df
    )


# =====================================================
# RESUMEN GLOBAL
# =====================================================

def build_summary_dataframe(
    reports
):
    """
    Convierte lista de reportes
    en DataFrame final.
    """

    if len(reports) == 0:

        return pd.DataFrame()

    return pd.DataFrame(
        reports
    )