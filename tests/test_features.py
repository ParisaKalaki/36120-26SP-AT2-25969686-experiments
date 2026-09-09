import numpy as np
import pandas as pd

from at2_experiments.features import add_log_transforms


def test_add_log_transforms_creates_new_column():
    df = pd.DataFrame({"precipitation": [0, 1, 10]})
    result = add_log_transforms(df, ["precipitation"])
    assert "precipitation_log" in result.columns
    assert np.isclose(result["precipitation_log"].iloc[0], 0.0)


def test_add_log_transforms_does_not_modify_original_column():
    df = pd.DataFrame({"precipitation": [0, 1, 10]})
    result = add_log_transforms(df, ["precipitation"])
    assert list(result["precipitation"]) == [0, 1, 10]
