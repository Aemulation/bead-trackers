import cupy

import pytest

from src.trackers.bead_tracker.quadratic_polynomial_fitter import (
    QuadraticPolynomialFitter,
)

COEFFICIENTS = (1, -1, 4)


def f(x):
    return COEFFICIENTS[0] * x**2 + COEFFICIENTS[1] * x + COEFFICIENTS[2]


@pytest.fixture
def mock_points() -> cupy.ndarray:
    num_beads = 2000

    return cupy.array([[f(x) for x in (0, 1, 2, 3, 4)]] * num_beads, dtype=cupy.float32)


@pytest.fixture
def quadratic_polynomial_fitter():
    weights = cupy.array([0.5, 0.85, 1.0, 0.85, 0.5])
    return QuadraticPolynomialFitter(weights)


def test_quadratic_polynomial_fitter(
    quadratic_polynomial_fitter: QuadraticPolynomialFitter, mock_points: cupy.ndarray
):
    coefficients = quadratic_polynomial_fitter.fit_2d(mock_points)

    for coefficient in coefficients:
        assert cupy.isclose(coefficient[0], COEFFICIENTS[0])
        assert cupy.isclose(coefficient[1], COEFFICIENTS[1])
        assert cupy.isclose(coefficient[2], COEFFICIENTS[2])


def test_get_top(
    quadratic_polynomial_fitter: QuadraticPolynomialFitter, mock_points: cupy.ndarray
):
    coefficients = quadratic_polynomial_fitter.fit_2d(mock_points)
    tops = quadratic_polynomial_fitter.get_top(coefficients)

    expected_top = -COEFFICIENTS[1] / (2 * COEFFICIENTS[0])

    for top in tops:
        assert cupy.isclose(top, expected_top)


def test_quadratic_polynomial_fitter_time(
    quadratic_polynomial_fitter: QuadraticPolynomialFitter, mock_points: cupy.ndarray
):
    weights = cupy.array([0.5, 0.85, 1.0, 0.85, 0.5])
    quadratic_polynomial_fitter = QuadraticPolynomialFitter(weights)

    total_elapsed = 0
    first = True
    num_iters = 1000
    s2 = cupy.cuda.Stream()
    for _ in range(num_iters):
        e1 = cupy.cuda.Event()
        e1.record()
        coefficients = quadratic_polynomial_fitter.fit_2d(mock_points)
        e2 = cupy.cuda.get_current_stream().record()

        # set up a stream order
        s2.wait_event(e2)
        with s2:
            # the a_cp is guaranteed updated when this copy (on s2) starts
            coefficients = cupy.asnumpy(coefficients)

        # timing
        e2.synchronize()
        t = cupy.cuda.get_elapsed_time(
            e1, e2
        )  # only include the compute time, not the copy time
        if not first:
            total_elapsed += t
        first = False
        print(f"ELAPSED: {t}ms")

    print(f"AVERAGE ELAPSED: {total_elapsed / (num_iters - 1)}ms")
