import cupy
import time


class QuadraticPolynomialFitter:
    def __init__(
        self,
        weights: cupy.ndarray = cupy.array([0.15, 0.5, 0.85, 1.0, 0.85, 0.5, 0.15]),
    ) -> None:
        # self.__square_root_weights = cupy.sqrt(weights)
        self.__square_root_weights = weights
        self.__x_matrix = self.__make_x_matrix()

    def __make_x_matrix(self) -> cupy.ndarray:
        num_points = self.__square_root_weights.shape[0]

        x_points = cupy.arange(num_points)

        column1 = cupy.square(x_points)
        column2 = x_points
        column3 = cupy.ones_like(x_points)

        unweighted_x_matrix = cupy.vstack((column1, column2, column3))
        weighted_x_matrix = (
            unweighted_x_matrix * self.__square_root_weights[cupy.newaxis, :]
        )
        return weighted_x_matrix.copy()

    def fit_2d(self, points_table: cupy.ndarray) -> cupy.ndarray:
        assert points_table.dtype == cupy.float32

        num_points = points_table.shape[1]
        assert num_points == self.__x_matrix.shape[1]

        weighted_points = points_table * self.__square_root_weights[cupy.newaxis, :]

        # start = time.perf_counter()
        # (coefficients, _, _, _) = cupy.linalg.lstsq(
        #     self.__x_matrix.T,
        #     weighted_points.T,
        # )
        # return coefficients.T

        coefficients = cupy.linalg.solve(
            self.__x_matrix @ self.__x_matrix.T, self.__x_matrix @ weighted_points.T
        ).T
        # end = time.perf_counter()
        # print(f"least sqaures took {end - start} seconds on cpu")
        return coefficients

    def get_top(self, coefficients: cupy.ndarray) -> cupy.ndarray:
        return -coefficients[:, 1:2] / (2 * coefficients[:, 0:1])
