"""Sequential subsampling logistic regression benchmark artifact."""

from ifsn_logistic.estimator import IFSNLogisticRegression
from ifsn_logistic.solver_reference import ifsn_reference_solve
from ifsn_logistic.solver_safe import ifsn_safe_solve

__version__ = "0.2.0"

__all__ = [
    "IFSNLogisticRegression",
    "ifsn_reference_solve",
    "ifsn_safe_solve",
]
