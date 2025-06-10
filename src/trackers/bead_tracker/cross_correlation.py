import cupy


def cross_correlate_nested_1d(
    input1: cupy.ndarray, input2: cupy.ndarray
) -> cupy.ndarray:
    length = input1.shape[1]
    assert input1.shape[1] == input2.shape[1]

    padded_input1 = cupy.pad(input1, ((0, 0), (0, length - 1)))
    padded_input2 = cupy.pad(input2, ((0, 0), (0, length - 1)))
    assert padded_input1.shape[1] == padded_input2.shape[1]

    in1 = cupy.fft.fft(padded_input1, axis=1)
    in2 = cupy.conj(cupy.fft.fft(padded_input2, axis=1))

    cross_correlation_result = cupy.fft.ifft(cupy.multiply(in1, in2), axis=1).real

    return cupy.roll(cross_correlation_result, (length // 2), axis=1)[:, :length].copy()
