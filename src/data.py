"""Load and preprocess the UCI Mushroom dataset."""

import pandas as pd
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42
DROP_COLUMNS = ["veil-type"]


def load_raw():
    """Download the raw UCI Mushroom dataset."""
    ds = fetch_ucirepo(id=73)
    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0].copy()
    return X, y


def preprocess(X, y):
    """Clean and one-hot encode the mushroom dataset."""

    # Remove the constant column if it exists.
    X = X.drop(columns=[c for c in DROP_COLUMNS if c in X.columns])

    # Replace missing values with an explicit category.
    X = X.fillna("unknown")

    # Convert categorical columns into numeric/boolean columns.
    X = pd.get_dummies(X)

    # Poisonous = 1, edible = 0.
    y = (y == "p").astype(int)

    return X, y


def load_mushroom():
    """Load and preprocess the mushroom dataset."""
    return preprocess(*load_raw())