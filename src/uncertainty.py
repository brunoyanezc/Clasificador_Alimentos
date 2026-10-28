import pandas as pd

UNCERTAINTY_PATH = (
    "data/uncertainty_defaults.csv"
)


def load_uncertainties():

    return pd.read_csv(
        UNCERTAINTY_PATH
    )


def uncertainty_dict():

    df = load_uncertainties()

    return dict(
        zip(
            df["Analyte"],
            df["CV"]
        )
    )