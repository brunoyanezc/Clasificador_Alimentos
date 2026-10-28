import pandas as pd

from src.config import (
    METADATA_COLUMNS,
    COLUMN_ALIASES
)

EXCEL_PATH = "data/specifications.xlsx"


def standardize_columns(df):

    renamed = {}

    for col in df.columns:

        col_clean = str(col).strip()

        if col_clean in COLUMN_ALIASES:

            renamed[col] = COLUMN_ALIASES[col_clean]

    df = df.rename(
        columns=renamed
    )

    return df


def get_client_names():

    workbook = pd.ExcelFile(
        EXCEL_PATH
    )

    return workbook.sheet_names


def validate_required_columns(df):

    missing = []

    for col in METADATA_COLUMNS:

        if col not in df.columns:

            missing.append(col)

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def validate_duplicates(df):

    duplicated = df.duplicated(
        subset=[
            "Calibre",
            "Product_Name"
        ]
    )

    if duplicated.any():

        dup_rows = df.loc[
            duplicated,
            [
                "Calibre",
                "Product_Name"
            ]
        ]

        raise ValueError(
            "Duplicated rows found:\n"
            f"{dup_rows}"
        )


def load_client_sheet(
    client_name
):

    df = pd.read_excel(
        EXCEL_PATH,
        sheet_name=client_name
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    df = standardize_columns(
        df
    )

    validate_required_columns(
        df
    )

    df["Calibre"] = (
        df["Calibre"]
        .astype(str)
        .str.strip()
    )

    validate_duplicates(
        df
    )

    return df


def get_calibres(df):

    return sorted(
        df["Calibre"]
        .astype(str)
        .unique()
    )


def get_client_size_subset(
    df,
    calibre
):

    calibre = str(
        calibre
    )

    subset = df[
        df["Calibre"]
        .astype(str)
        ==
        calibre
    ].copy()

    subset.reset_index(
        drop=True,
        inplace=True
    )

    return subset


def get_product_names(df):

    return sorted(
        df["Product_Name"]
        .unique()
    )


def get_families(df):

    return sorted(
        df["Family"]
        .dropna()
        .unique()
    )