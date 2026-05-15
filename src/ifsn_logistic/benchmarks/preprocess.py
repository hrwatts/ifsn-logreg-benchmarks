"""Preprocessing utilities used by the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DenseOneHotEncoder:
    """Simple deterministic one-hot encoder with dense NumPy output."""

    categories_: list[np.ndarray] | None = None
    feature_names_: list[str] | None = None

    def fit(self, X: np.ndarray, column_names: list[str]) -> "DenseOneHotEncoder":
        X = np.asarray(X, dtype=object)
        if X.ndim != 2:
            raise ValueError("DenseOneHotEncoder expects a 2D array.")

        categories: list[np.ndarray] = []
        feature_names: list[str] = []
        for idx, name in enumerate(column_names):
            cats = np.array(sorted({str(v) for v in X[:, idx].tolist()}), dtype=object)
            categories.append(cats)
            feature_names.extend([f"{name}={cat}" for cat in cats.tolist()])

        self.categories_ = categories
        self.feature_names_ = feature_names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.categories_ is None:
            raise RuntimeError("DenseOneHotEncoder must be fit before transform.")

        X = np.asarray(X, dtype=object)
        if X.ndim != 2:
            raise ValueError("DenseOneHotEncoder expects a 2D array.")
        if X.shape[1] != len(self.categories_):
            raise ValueError("DenseOneHotEncoder received the wrong number of columns.")

        encoded_blocks: list[np.ndarray] = []
        for idx, cats in enumerate(self.categories_):
            column = X[:, idx].astype(str)
            block = np.zeros((X.shape[0], len(cats)), dtype=np.float64)
            cat_to_pos = {cat: pos for pos, cat in enumerate(cats.tolist())}
            for row_idx, value in enumerate(column.tolist()):
                pos = cat_to_pos.get(value)
                if pos is not None:
                    block[row_idx, pos] = 1.0
            encoded_blocks.append(block)

        return np.hstack(encoded_blocks) if encoded_blocks else np.empty((X.shape[0], 0))


@dataclass
class DenseStandardScaler:
    """Dense column-wise standardization with safe zero-variance handling."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DenseStandardScaler":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("DenseStandardScaler expects a 2D array.")

        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("DenseStandardScaler must be fit before transform.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_
