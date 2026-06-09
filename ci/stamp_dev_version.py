"""Stamp a derived dev version into pyproject.toml + Cargo manifests.

Run only in CI on main-branch builds. The patched files are consumed by the
downstream build jobs but never committed.

Python (PEP 440):   <base>.dev<N>+g<sha>      e.g. 0.2.0.dev42+gabc1234
Cargo  (SemVer):    <base>-dev.<N>            e.g. 0.2.0-dev.42

<N> is the total commit count on HEAD (monotonic, doesn't depend on tag fetch).

Three files are patched:
- pyproject.toml             [project].version  -> Python dev string
- Cargo.toml                 [workspace.package].version -> Cargo dev string
- crates/rustamer/Cargo.toml [package].version (was `version.workspace = true`)

The third patch is necessary because maturin's wheel-naming logic does not
always follow Cargo's `workspace.true` inheritance in cross-compile / container
build contexts; replacing the inheritance line with a literal version makes
the wheel name pick up the dev string reliably.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import tomllib


def sh(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    base = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"][
        "version"
    ]
    if not re.match(r"^\d+\.\d+\.\d+$", base):
        sys.exit(f"refusing to stamp on top of an already-suffixed version: {base!r}")

    n = sh("git", "rev-list", "--count", "HEAD")
    sha = sh("git", "rev-parse", "--short=7", "HEAD")

    py_v = f"{base}.dev{n}+g{sha}"
    rs_v = f"{base}-dev.{n}"

    py = pathlib.Path("pyproject.toml")
    py.write_text(
        re.sub(
            r'^version = ".*"',
            f'version = "{py_v}"',
            py.read_text(),
            count=1,
            flags=re.M,
        )
    )

    cg = pathlib.Path("Cargo.toml")
    cg.write_text(
        re.sub(
            r'^version = ".*"',
            f'version = "{rs_v}"',
            cg.read_text(),
            count=1,
            flags=re.M,
        )
    )

    # Shadow `version.workspace = true` in the rustamer crate with a literal,
    # so maturin's wheel naming picks up the dev version even when the
    # build runs inside a manylinux container that doesn't resolve the
    # workspace inheritance for naming.
    rc = pathlib.Path("crates/rustamer/Cargo.toml")
    rc.write_text(
        re.sub(
            r"^version\.workspace = true$",
            f'version = "{rs_v}"',
            rc.read_text(),
            count=1,
            flags=re.M,
        )
    )

    print(f"stamped pyproject={py_v} cargo={rs_v}")


if __name__ == "__main__":
    main()
