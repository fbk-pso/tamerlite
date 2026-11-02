# Copyright (C) 2025 PSO Unit, Fondazione Bruno Kessler
# This file is part of TamerLite.
#
# TamerLite is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TamerLite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

"""Tests for `tamerlite.core`'s backend selection (`DISABLE_RUSTAMER`)."""

import os
import subprocess
import sys

import pytest

from tamerlite.core import _rustamer_disabled


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " true", "On\n"])
def test_disable_rustamer_true_values(value: str):
    assert _rustamer_disabled(value)


@pytest.mark.parametrize(
    "value", [None, "", "  ", "0", "false", "False", "no", "off", " OFF "]
)
def test_disable_rustamer_false_values(value: str | None):
    assert not _rustamer_disabled(value)


@pytest.mark.parametrize("value", ["2", "maybe", "ture"])
def test_disable_rustamer_invalid_value_warns_and_keeps_default(value: str):
    with pytest.warns(UserWarning, match="DISABLE_RUSTAMER"):
        assert not _rustamer_disabled(value)


@pytest.mark.parametrize("value", ["", "maybe"])
def test_disable_rustamer_import_does_not_exit(value: str):
    # Exercises the import-time path itself, which a reload in this process
    # would leave half-applied for the rest of the session.
    env = {**os.environ, "DISABLE_RUSTAMER": value}
    proc = subprocess.run(
        [sys.executable, "-c", "import tamerlite.core"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
