import numpy as np


def dataframe_to_vector(
    row,
    analytes
):

    return (
        row[analytes]
        .astype(float)
        .values
    )


def uncertainty_dict_to_vector(
    analytes,
    uncertainty_dict
):

    return np.array(
        [
            uncertainty_dict[a]
            for a in analytes
        ]
    )