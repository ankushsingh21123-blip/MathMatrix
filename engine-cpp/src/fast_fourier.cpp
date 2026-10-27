#include "fast_fourier.hpp"
#include <cmath>
#include <numbers>

namespace mathmatrix::fft {

const double PI = 3.14159265358979323846;

void FastFourierTransform::fft(std::vector<Complex>& a) {
    size_t n = a.size();
    if (n <= 1) return;

    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    // Cooley-Tukey Radix-2 Butterfly Computation
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * PI / static_cast<double>(len);
        Complex wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                Complex u = a[i + j];
                Complex v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

void FastFourierTransform::ifft(std::vector<Complex>& a) {
    for (auto& x : a) {
        x = std::conj(x);
    }
    fft(a);
    for (auto& x : a) {
        x = std::conj(x) / static_cast<double>(a.size());
    }
}

std::vector<Complex> FastFourierTransform::naive_dft(const std::vector<Complex>& x) {
    size_t n = x.size();
    std::vector<Complex> out(n);
    for (size_t k = 0; k < n; ++k) {
        Complex sum(0.0, 0.0);
        for (size_t t = 0; t < n; ++t) {
            double angle = -2.0 * PI * static_cast<double>(k * t) / static_cast<double>(n);
            sum += x[t] * Complex(std::cos(angle), std::sin(angle));
        }
        out[k] = sum;
    }
    return out;
}

std::vector<std::vector<double>> FastFourierTransform::matrix_multiply(
    const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b
) {
    size_t rA = a.size();
    size_t cA = a[0].size();
    size_t cB = b[0].size();

    std::vector<std::vector<double>> res(rA, std::vector<double>(cB, 0.0));
    for (size_t i = 0; i < rA; ++i) {
        for (size_t k = 0; k < cA; ++k) {
            for (size_t j = 0; j < cB; ++j) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
}

} // namespace mathmatrix::fft
