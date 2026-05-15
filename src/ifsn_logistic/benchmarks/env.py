"""Environment provenance capture for benchmark runs."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from ifsn_logistic import __version__


def _safe_import_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Portable runtime snapshot for reproducibility records."""

    timestamp_utc: str
    os_name: str
    os_release: str
    os_version: str
    machine: str
    processor: str
    cpu_count: int | None
    python_version: str
    python_implementation: str
    numpy_version: str | None
    scipy_version: str | None
    sklearn_version: str | None
    pandas_version: str | None
    ifsn_version: str

    @staticmethod
    def capture() -> "RuntimeEnvironment":
        return RuntimeEnvironment(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            os_name=platform.system(),
            os_release=platform.release(),
            os_version=platform.version(),
            machine=platform.machine(),
            processor=platform.processor(),
            cpu_count=os.cpu_count(),
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
            numpy_version=_safe_import_version("numpy"),
            scipy_version=_safe_import_version("scipy"),
            sklearn_version=_safe_import_version("sklearn"),
            pandas_version=_safe_import_version("pandas"),
            ifsn_version=__version__,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)