"""Fail CI if the version manifests report different base versions.

The base version is the leading ``X.Y.Z`` triple. Pre-release / local suffixes
(``.devN``, ``+gsha``, ``-dev.N``) are ignored so dev-stamped builds still pass.

Three manifests must agree:
- ``pyproject.toml``                 -> ``[project].version``
- ``Cargo.toml``                     -> ``[workspace.package].version``
- ``crates/rustamer/pyproject.toml`` -> ``[project].version``
"""

from __future__ import annotations

import pathlib
import re
import sys

import tomllib


def base(version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        sys.exit(f"unparseable version: {version!r}")
    return ".".join(match.groups())


def main() -> None:
    py = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"]
    rs = tomllib.loads(pathlib.Path("Cargo.toml").read_text())["workspace"]["package"][
        "version"
    ]
    wheel = tomllib.loads(pathlib.Path("crates/rustamer/pyproject.toml").read_text())[
        "project"
    ]["version"]
    if not base(py) == base(rs) == base(wheel):
        sys.exit(
            f"version base mismatch: pyproject={py} cargo={rs} rustamer-wheel={wheel}"
        )
    print(f"OK: pyproject={py} cargo={rs} rustamer-wheel={wheel}")


if __name__ == "__main__":
    main()
