import numpy as np
import pandas as pd
import streamlit as st

from src.loader import (
    get_client_names,
    load_client_sheet,
    get_calibres,
    get_client_size_subset
)

from src.uncertainty import (
    uncertainty_dict
)

from src.analytes import (
    get_all_analytes,
    get_active_analytes
)

from src.classifier import (
    classify_sample
)

from src.diagnostics import (
    analyte_contributions,
    coverage_ratio,
    diagnostics_metadata,
    build_job_report
)

from src.pca_model import (
    run_pca,
    project_multiple_samples,
    get_pca_summary
)

from src.plots import (
    plot_pca,
    plot_top_contributors,
    plot_probability_ranking,
    plot_distance_ranking
)

# ==========================================================
# PAGE
# ==========================================================

st.set_page_config(
    page_title="Matriz Target Alimentos",
    layout="wide"
)

st.title(
    "Matriz Target alimentos"
)

# ==========================================================
# CLIENTE
# ==========================================================

clients = get_client_names()

selected_client = st.sidebar.selectbox(
    "Cliente",
    clients
)

client_df = load_client_sheet(
    selected_client
)

# ==========================================================
# CALIBRE
# ==========================================================

calibres = get_calibres(
    client_df
)

selected_calibre = st.sidebar.selectbox(
    "Calibre",
    calibres
)

working_df = get_client_size_subset(
    client_df,
    selected_calibre
)

# ==========================================================
# METRICA
# ==========================================================

metric_label = st.sidebar.selectbox(
    "Métrica",
    [
        "Euclidean",
        "Mahalanobis"
    ]
)

metric = {
    "Euclidean": "euclidean",
    "Mahalanobis": "mahalanobis"
}[metric_label]

# ==========================================================
# PCA COLORING
# ==========================================================

pca_color_mode = st.sidebar.radio(
    "Color PCA",
    [
        "Family",
        "Product_Name"
    ]
)

# ==========================================================
# INCERTIDUMBRES
# ==========================================================

unc_dict = uncertainty_dict()

with st.sidebar.expander(
    "Incertidumbres",
    expanded=False
):

    for analyte in sorted(
        unc_dict.keys()
    ):

        unc_dict[analyte] = st.number_input(
            analyte,
            value=float(
                unc_dict[analyte]
            ),
            min_value=0.00001,
            format="%.6f"
        )

# ==========================================================
# ANALITOS
# ==========================================================

available_analytes = (
    get_all_analytes(
        working_df
    )
)

st.subheader(
    "Analitos"
)

selected_analytes = st.multiselect(
    "Seleccionar analitos",
    available_analytes,
    default=available_analytes
)

if not selected_analytes:

    st.warning(
        "Seleccione al menos un analito."
    )

    st.stop()

# ==========================================================
# TABLA ENTRADA
# ==========================================================

st.subheader(
    "Ingreso de muestras"
)

column_config = {
    "JOB":
        st.column_config.TextColumn(
            "JOB"
        )
}

for analyte in selected_analytes:

    column_config[
        analyte
    ] = st.column_config.NumberColumn(
        analyte,
        format="%.4f"
    )

initial_df = pd.DataFrame(
    {
        "JOB": ["JOB001"]
    }
)

for analyte in selected_analytes:

    initial_df[
        analyte
    ] = np.nan

sample_df = st.data_editor(
    initial_df,
    column_config=column_config,
    num_rows="dynamic",
    use_container_width=True
)

# ==========================================================
# CLASIFICAR
# ==========================================================

run_button = st.button(
    "Clasificar"
)

if not run_button:

    st.stop()

# ==========================================================
# PCA
# ==========================================================

try:

    pca_analytes = (
        get_active_analytes(
            working_df,
            unc_dict
        )
    )

    (
        pca,
        scaler,
        pca_df,
        explained,
        loadings
    ) = run_pca(
        working_df,
        pca_analytes
    )

    pca_summary = get_pca_summary(
        explained
    )

    pca_available = True

except Exception:

    pca_available = False

# ==========================================================
# LOOP JOBS
# ==========================================================

reports = []

diagnostics_store = {}

ranking_store = {}

for _, sample_row in sample_df.iterrows():

    job = str(
        sample_row["JOB"]
    )

    user_analytes = []

    for analyte in selected_analytes:

        value = sample_row[
            analyte
        ]

        if pd.notna(value):

            user_analytes.append(
                analyte
            )

    if len(user_analytes) == 0:

        continue

    active_analytes = (
        get_active_analytes(
            working_df,
            unc_dict,
            user_columns=user_analytes
        )
    )

    if len(active_analytes) == 0:

        continue

    ranking = classify_sample(
        sample_row=sample_row,
        centroids_df=working_df,
        active_analytes=active_analytes,
        uncertainty_dict=unc_dict,
        metric=metric
    )

    winner = ranking.iloc[0]

    winner_row = working_df[
        working_df[
            "Product_Name"
        ]
        ==
        winner[
            "Product_Name"
        ]
    ].iloc[0]

    diagnostics_df = (
        analyte_contributions(
            sample_row,
            winner_row,
            active_analytes,
            unc_dict
        )
    )

    coverage = coverage_ratio(
        active_analytes,
        selected_analytes
    )

    report = build_job_report(
        job,
        ranking,
        diagnostics_df,
        coverage
    )

    reports.append(
        report
    )

    diagnostics_store[
        job
    ] = diagnostics_df

    ranking_store[
        job
    ] = ranking

# ==========================================================
# RESULTADOS
# ==========================================================

if len(reports) == 0:

    st.warning(
        "No existen muestras clasificables."
    )

    st.stop()

results_df = pd.DataFrame(
    reports
)

st.subheader(
    "Resultados"
)

st.dataframe(
    results_df,
    use_container_width=True
)

# ==========================================================
# JOB
# ==========================================================

selected_job = st.selectbox(
    "JOB",
    results_df["JOB"]
)

diagnostics_df = (
    diagnostics_store[
        selected_job
    ]
)

ranking_df = (
    ranking_store[
        selected_job
    ]
)

# ==========================================================
# METADATA
# ==========================================================

metadata = diagnostics_metadata(
    selected_analytes,
    diagnostics_df[
        "Analyte"
    ].tolist()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Seleccionados",
    metadata["Selected"]
)

c2.metric(
    "Utilizados",
    metadata["Used"]
)

c3.metric(
    "Cobertura",
    f"{metadata['Coverage']:.1%}"
)

c4.metric(
    "Calidad",
    metadata["Coverage_Quality"]
)

# ==========================================================
# RANKING
# ==========================================================

st.subheader(
    "Ranking"
)

st.dataframe(
    ranking_df,
    use_container_width=True
)

col1, col2 = st.columns(2)

with col1:

    fig = plot_probability_ranking(
        ranking_df
    )

    st.pyplot(fig)

with col2:

    fig = plot_distance_ranking(
        ranking_df
    )

    st.pyplot(fig)

# ==========================================================
# DIAGNOSTICO
# ==========================================================

st.subheader(
    "Contribución de analitos"
)

st.dataframe(
    diagnostics_df,
    use_container_width=True
)

fig = plot_top_contributors(
    diagnostics_df
)

st.pyplot(fig)

# ==========================================================
# PCA
# ==========================================================

if pca_available:

    st.subheader(
        "PCA"
    )

    try:

        valid_rows = sample_df.dropna(
            subset=pca_analytes
        )

        projected = (
            project_multiple_samples(
                valid_rows,
                pca_analytes,
                scaler,
                pca
            )
        )

    except Exception:

        projected = None

    fig = plot_pca(
        pca_df,
        projected
    )

    st.pyplot(fig)

    st.write(
        f"PC1 = {pca_summary['PC1_Variance_%']:.1f}%"
    )

    st.write(
        f"PC2 = {pca_summary['PC2_Variance_%']:.1f}%"
    )

    st.subheader(
        "Loadings"
    )

    st.dataframe(
        loadings,
        use_container_width=True
    )