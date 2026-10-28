import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =====================================================
# VALIDACIONES
# =====================================================

def validate_pca_input(
    working_df,
    active_analytes
):
    """
    Validaciones mínimas para PCA.
    """

    if len(active_analytes) < 2:

        raise ValueError(
            "PCA requiere al menos "
            "2 analitos."
        )

    if len(working_df) < 3:

        raise ValueError(
            "PCA requiere al menos "
            "3 Product_Name."
        )

    missing = []

    for analyte in active_analytes:

        if analyte not in working_df.columns:

            missing.append(
                analyte
            )

    if missing:

        raise ValueError(
            "Analitos faltantes: "
            + ", ".join(missing)
        )


# =====================================================
# MATRIZ PCA
# =====================================================

def build_pca_matrix(
    working_df,
    active_analytes
):
    """
    Construye matriz PCA.
    """

    X = (
        working_df[
            active_analytes
        ]
        .astype(float)
        .copy()
    )

    return X


# =====================================================
# PCA PRINCIPAL
# =====================================================

def run_pca(
    working_df,
    active_analytes
):
    """
    Ejecuta PCA para el subconjunto
    Cliente + Calibre.
    """

    validate_pca_input(
        working_df,
        active_analytes
    )

    X = build_pca_matrix(
        working_df,
        active_analytes
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    pca = PCA(
        n_components=2
    )

    coordinates = pca.fit_transform(
        X_scaled
    )

    pca_df = pd.DataFrame(
        {
            "PC1":
                coordinates[:, 0],

            "PC2":
                coordinates[:, 1],

            "Product_Name":
                working_df[
                    "Product_Name"
                ].values,

            "Family":
                working_df[
                    "Family"
                ].values

        }
    )

    explained_variance = (
        pca.explained_variance_ratio_
    )

    loadings = pd.DataFrame(
        {
            "Analyte":
                active_analytes,

            "PC1":
                pca.components_[0],

            "PC2":
                pca.components_[1]
        }
    )

    return (
        pca,
        scaler,
        pca_df,
        explained_variance,
        loadings
    )


# =====================================================
# PROYECCION INDIVIDUAL
# =====================================================

def project_sample(
    sample_row,
    active_analytes,
    scaler,
    pca
):
    """
    Proyecta una muestra individual.
    """

    missing_values = []

    for analyte in active_analytes:

        if pd.isna(
            sample_row.get(
                analyte,
                np.nan
            )
        ):
            missing_values.append(
                analyte
            )

    if missing_values:

        raise ValueError(
            "Faltan analitos para PCA: "
            + ", ".join(missing_values)
        )

    x = (
        sample_row[
            active_analytes
        ]
        .astype(float)
        .values
    )

    x = x.reshape(
        1,
        -1
    )

    x_scaled = scaler.transform(
        x
    )

    coords = pca.transform(
        x_scaled
    )

    return {
        "PC1": coords[0, 0],
        "PC2": coords[0, 1]
    }


# =====================================================
# VALIDACION JOB PCA
# =====================================================

def is_sample_projectable(
    sample_row,
    active_analytes
):
    """
    Determina si una muestra puede
    proyectarse al PCA.
    """

    for analyte in active_analytes:

        if analyte not in sample_row:

            return False

        if pd.isna(
            sample_row[
                analyte
            ]
        ):
            return False

    return True


# =====================================================
# PROYECCION MASIVA
# =====================================================

def project_multiple_samples(
    samples_df,
    active_analytes,
    scaler,
    pca
):
    """
    Proyección masiva de JOBs.
    """

    results = []

    for _, row in samples_df.iterrows():

        if not is_sample_projectable(
            row,
            active_analytes
        ):
            continue

        try:

            projection = (
                project_sample(
                    row,
                    active_analytes,
                    scaler,
                    pca
                )
            )

            results.append(
                {
                    "JOB":
                        row["JOB"],

                    "PC1":
                        projection["PC1"],

                    "PC2":
                        projection["PC2"]
                }
            )

        except Exception:

            continue

    return pd.DataFrame(
        results
    )


# =====================================================
# JOBS NO PROYECTABLES
# =====================================================

def get_non_projectable_jobs(
    samples_df,
    active_analytes
):
    """
    Retorna JOBs que no pueden
    proyectarse al PCA.
    """

    jobs = []

    for _, row in samples_df.iterrows():

        if not is_sample_projectable(
            row,
            active_analytes
        ):
            jobs.append(
                row["JOB"]
            )

    return jobs


# =====================================================
# RESUMEN PCA
# =====================================================

def get_pca_summary(
    explained_variance
):
    """
    Resumen amigable.
    """

    return {
        "PC1_Variance_%":
            round(
                explained_variance[0]
                * 100,
                2
            ),

        "PC2_Variance_%":
            round(
                explained_variance[1]
                * 100,
                2
            ),

        "Total_Variance_%":
            round(
                (
                    explained_variance[0]
                    +
                    explained_variance[1]
                )
                * 100,
                2
            )
    }