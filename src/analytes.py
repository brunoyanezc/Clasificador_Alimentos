from src.config import (
    METADATA_COLUMNS
)


def get_all_analytes(df):

    return [

        c

        for c in df.columns

        if c not in METADATA_COLUMNS

    ]


def get_active_analytes(
    working_df,
    uncertainty_dict,
    user_columns=None
):

    active = []

    candidate_columns = (
        get_all_analytes(
            working_df
        )
    )

    for analyte in candidate_columns:

        if analyte not in uncertainty_dict:

            continue

        if working_df[
            analyte
        ].isna().any():

            continue

        try:

            if (
                working_df[
                    analyte
                ].nunique()
                <= 1
            ):
                continue

        except Exception:

            continue

        if user_columns is not None:

            if analyte not in user_columns:

                continue

        active.append(
            analyte
        )

    return active