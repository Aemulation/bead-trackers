import cupy
import time


class QuadraticPolynomialFitter:
    def __init__(
        self,
        num_batches: int,
        weights: cupy.ndarray = cupy.array([0.15, 0.5, 0.85, 1.0, 0.85, 0.5, 0.15]),
    ) -> None:
        self.__num_batches = num_batches
        self.__weights = weights
        self.__x_matrix = self.__make_x_matrix(num_batches, weights)

    def __make_x_matrix(self, num_batches: int, weights: cupy.ndarray) -> cupy.ndarray:
        num_points = weights.shape[0]

        x_points = cupy.arange(num_points)

        column1 = cupy.square(x_points)
        column2 = x_points
        column3 = cupy.ones_like(x_points)

        unweighted_x_matrix = cupy.vstack((column1, column2, column3))
        return cupy.repeat(
            cupy.expand_dims(unweighted_x_matrix, axis=0), repeats=num_batches, axis=0
        )

    def __batched_least_squares(
        self, X: cupy.ndarray, Y: cupy.ndarray, weights: cupy.ndarray
    ) -> cupy.ndarray:
        W = cupy.diag(weights)

        XtWX = cupy.einsum("bij,ii,bik->bjk", X, W, X)
        XtWX_inv = cupy.linalg.inv(XtWX)

        XtWY = cupy.einsum("bij,ii, bi->bj", X, W, Y)

        return cupy.einsum("bij,bj->bi", XtWX_inv, XtWY)

    def fit_2d(self, points_table: cupy.ndarray) -> cupy.ndarray:
        assert points_table.dtype == cupy.float32

        assert points_table.shape[0] == self.__num_batches

        num_points = points_table.shape[1]
        assert num_points == self.__x_matrix.shape[2]

        coefficients = self.__batched_least_squares(
            cupy.transpose(self.__x_matrix, (0, 2, 1)), points_table, self.__weights
        )

        return coefficients

    def get_top(self, coefficients: cupy.ndarray) -> cupy.ndarray:
        return -coefficients[:, 1:2] / (2 * coefficients[:, 0:1])
