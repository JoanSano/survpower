"""Deprecation of survpower.report (#8) and its replacement for Table 8."""

import importlib
import subprocess
import sys

import numpy as np
import pytest

from survpower.concordance import mapping_table


def test_import_report_warns():
    sys.modules.pop("survpower.report", None)  # the warning fires on first import only
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        importlib.import_module("survpower.report")


def test_import_package_does_not_warn():
    res = subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c",
                          "import survpower, survpower.cli"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_mapping_table_matches_table8():
    df = mapping_table()
    assert np.round(df["beta"], 3).tolist() == [0.179, 0.370, 0.585, 0.844, 1.177]
    assert df["hr"].round(2).tolist() == [1.20, 1.45, 1.80, 2.33, 3.24]
    assert df["harrell_limit"].round(4).tolist() == [0.5501, 0.6006, 0.6517, 0.7032, 0.7548]
    assert df["N80_closed_form"].tolist() == [306, 72, 29, 14, 8]


def test_report_mapping_unchanged():
    with pytest.warns(DeprecationWarning):
        sys.modules.pop("survpower.report", None)
        report = importlib.import_module("survpower.report")
    df = report.run_cindex_mapping()
    assert df["beta"].tolist() == [0.0953, 0.1398, 0.1791, 0.3698, 0.5853, 0.8442, 1.1766]
    assert df["hr"].tolist() == [1.1, 1.15, 1.196, 1.447, 1.796, 2.326, 3.243]
    assert df["lam_c"].tolist() == [0.01152, 0.01149, 0.01144, 0.01109, 0.0105, 0.00954, 0.00812]
    assert df["N80_closed_form"].tolist() == [1081, 503, 306, 72, 29, 14, 8]
