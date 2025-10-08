import pandas as pd

from modules.data_preparation import data_preparation


def test_data_preparation_without_split(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "feature_1": [0.1, 0.2, 0.3],
            "feature_2": [1, 2, 3],
            "labels": ["calm", "happy", "sad"],
        }
    )

    # Run the function in a temporary working directory
    monkeypatch.chdir(tmp_path)

    X_df, Y_df, kf = data_preparation(data, train_test=False)

    assert isinstance(X_df, pd.DataFrame)
    assert isinstance(Y_df, pd.DataFrame)
    assert kf is None

    x_path = tmp_path / "out" / "X.csv"
    y_path = tmp_path / "out" / "Y.csv"

    assert x_path.exists()
    assert y_path.exists()

    saved_X = pd.read_csv(x_path)
    saved_Y = pd.read_csv(y_path)

    pd.testing.assert_frame_equal(saved_X, X_df.reset_index(drop=True))
    pd.testing.assert_frame_equal(saved_Y, Y_df.reset_index(drop=True))
