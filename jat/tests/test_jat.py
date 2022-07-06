"""
Unit and regression test for the jat package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import jat


def test_jat_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "jat" in sys.modules
