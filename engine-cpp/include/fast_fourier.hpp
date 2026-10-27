#pragma once

#include <vector>
#include <complex>
#include <cstdint>

namespace mathmatrix::fft {

using Complex = std::complex<double>;

class FastFourierTransform {
public:
    // In-place Cooley-Tukey Radix-2 FFT (O(N log N))
    static void fft(std::vector<Complex>& x);

    // In-place Inverse FFT
    static void ifft(std::vector<Complex>& x);

    // Naive O(N^2) Discrete Fourier Transform for verification & benchmarking
    static std::vector<Complex> naive_dft(const std::vector<Complex>& x);

    // High-performance matrix dot product
    static std::vector<std::vector<double>> matrix_multiply(
        const std::vector<std::vector<double>>& a,
        const std::vector<std::vector<double>>& b
    );
};

} // namespace mathmatrix::fft
