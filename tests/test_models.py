"""Tests for statistics functions within the Model layer."""

import pytest
import numpy as np
import numpy.testing as npt


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_max_zeros():
    """Test that max function works for an array of zeros."""
    from inflammation.models import daily_max

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_max_integers():
    """Test that max function works for an array of positive integers."""
    from inflammation.models import daily_max

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([5, 6])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_max_integers2():
    """Test that max function works for an array of positive and negative integers."""
    from inflammation.models import daily_max

    test_input = np.array([[-1, 2, -9],
                           [3, -4, 0],
                           [-5, -6, -6],
                           [0, 0, 0]])
    test_result = np.array([3, 2, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_min_zeros():
    """Test that min function works for an array of zeros."""
    from inflammation.models import daily_min

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_integers():
    """Test that min function works for an array of positive integers."""
    from inflammation.models import daily_min

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([1, 2])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_integers2():
    """Test that min function works for an array of positive and negative integers."""
    from inflammation.models import daily_min

    test_input = np.array([[-1, 2, -9],
                           [3, -4, 0],
                           [-5, -6, -6],
                           [0, 0, 0]])
    test_result = np.array([-5, -6, -9])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [5, 6]),
        ([[-1, 2, -9], [3, -4, 0], [-5, -6, -6], [0, 0, 0]], [3, 2, 0]),
    ])
def test_daily_max(test, expected):
    """Test max function works for array of zeros and positive/negative integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [1, 2]),
        ([[-1, 2, -9], [3, -4, 0], [-5, -6, -6], [0, 0, 0]], [-5, -6, -9]),
    ])
def test_daily_min(test, expected):
    """Test min function works for array of zeros and positive/negative integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))
