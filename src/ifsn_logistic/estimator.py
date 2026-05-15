"""A lightweight sklearn-style estimator for sequential subsampling logistic regression."""

from __future__ import annotations

from typing import Any

import numpy as np

from ifsn_logistic.config import SafeSolverConfig, SolverConfig
from ifsn_logistic.math_utils import logistic_probabilities
from ifsn_logistic.solver_reference import ifsn_reference_solve
from ifsn_logistic.solver_safe import ifsn_safe_solve


def _as_float_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix. Got shape {X.shape}.")
    return X


def _as_label_array(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"Expected a 1D label array. Got shape {y.shape}.")
    return y


class IFSNLogisticRegression:
    """Binary logistic regression using sequential subsampling Newton updates."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-6,
        subsample_size: int | None = None,
        sampling_scheme: str = "fresh",
        fit_intercept: bool = True,
        solver: str = "safe",
        step_size: float = 1.0,
        line_search: bool = True,
        max_step_norm: float = 10.0,
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.subsample_size = subsample_size
        self.sampling_scheme = sampling_scheme
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.step_size = step_size
        self.line_search = line_search
        self.max_step_norm = max_step_norm
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return init parameters in sklearn-style form."""
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "subsample_size": self.subsample_size,
            "sampling_scheme": self.sampling_scheme,
            "fit_intercept": self.fit_intercept,
            "solver": self.solver,
            "step_size": self.step_size,
            "line_search": self.line_search,
            "max_step_norm": self.max_step_norm,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> "IFSNLogisticRegression":
        """Set parameters in place and return self."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter {key!r}.")
            setattr(self, key, value)
        return self

    def _augment(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X_aug = np.hstack([X, ones])
            pw = np.ones(X_aug.shape[1])
            pw[-1] = 0.0
            return X_aug, pw
        return X, None

    def _build_config(self) -> SolverConfig | SafeSolverConfig:
        common = dict(
            max_iter=self.max_iter,
            tol=self.tol,
            C=self.C,
            subsample_size=self.subsample_size,
            sampling_scheme=self.sampling_scheme,
            fit_intercept=False,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        if self.solver == "reference":
            return SolverConfig(**common)
        if self.solver == "safe":
            return SafeSolverConfig(
                **common,
                step_size=self.step_size,
                line_search=self.line_search,
                max_step_norm=self.max_step_norm,
            )
        raise ValueError(f"Unknown solver {self.solver!r}. Use 'reference' or 'safe'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IFSNLogisticRegression":
        """Fit the model on binary data."""
        X = _as_float_array(X)
        y = _as_label_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                "IFSNLogisticRegression supports binary classification only. "
                f"Got {len(classes)} classes."
            )

        self.classes_ = classes
        y_bin = (y == classes[1]).astype(np.float64)

        X_aug, pen_weights = self._augment(X)
        config = self._build_config()

        if self.solver == "reference":
            beta, diag = ifsn_reference_solve(X_aug, y_bin, config, pen_weights)
        else:
            beta, diag = ifsn_safe_solve(X_aug, y_bin, config, pen_weights)

        if self.fit_intercept:
            self.coef_ = beta[:-1].reshape(1, -1)
            self.intercept_ = np.array([beta[-1]])
        else:
            self.coef_ = beta.reshape(1, -1)
            self.intercept_ = np.array([0.0])

        self.n_features_in_ = X.shape[1]
        self.n_iter_ = diag["n_iter"]
        self.converged_ = diag["converged"]
        self.diagnostics_ = diag
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise AttributeError("This IFSNLogisticRegression instance is not fitted yet.")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw linear scores."""
        self._require_fitted()
        X = _as_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features.")
        return X @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        self._require_fitted()
        X = _as_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features.")
        X_aug, _ = self._augment(X)
        if self.fit_intercept:
            beta = np.concatenate([self.coef_.ravel(), self.intercept_])
        else:
            beta = self.coef_.ravel()
        p1 = logistic_probabilities(X_aug, beta)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        indices = proba.argmax(axis=1)
        return self.classes_[indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean classification accuracy."""
        y = _as_label_array(y)
        pred = self.predict(X)
        if pred.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")
        return float(np.mean(pred == y))
