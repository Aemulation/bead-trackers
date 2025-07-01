import cupy


def cross_correlate_nested_1d(input: cupy.ndarray) -> cupy.ndarray:
    length = input.shape[1]

    padded_input1 = cupy.pad(input, ((0, 0), (0, length - 1)))
    padded_input2 = cupy.pad(cupy.flip(input, axis=0), ((0, 0), (0, length - 1)))

    input1 = cupy.fft.fft(padded_input1, axis=1)
    input2 = cupy.conj(cupy.fft.fft(padded_input2, axis=1))

    cross_correlation_result = cupy.fft.ifft(cupy.multiply(input1, input2), axis=1).real

    return cupy.roll(cross_correlation_result, (length // 2), axis=1)[:, :length].copy()
