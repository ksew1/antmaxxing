"""Unit tests for the antmaxxing operations module."""

from antmaxxing import add


def test_add() -> None:
    """Test that add() returns correct sum."""
    expected = 5
    assert add(2, 3) == expected
