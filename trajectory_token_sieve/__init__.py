"""Local checkout shim for direct `python -m trajectory_token_sieve...`.

The installable package lives under src/. This file makes the package visible
when commands are run from a fresh checkout without `pip install -e .`.
"""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "trajectory_token_sieve"
__path__ = [str(_SRC_PACKAGE)]
__file__ = str(_SRC_PACKAGE / "__init__.py")
exec((_SRC_PACKAGE / "__init__.py").read_text(encoding="utf-8"), globals())
