#include "fast_fourier.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

int main() {
    std::cout << "========================================================\n";
    std::cout << "    MATHMATRIX C++20 FAST FOURIER TRANSFORM ENGINE     \n";
    std::cout << "========================================================\n\n";

    constexpr size_t N = 1024; // 1024-point complex FFT
    std::cout << "[*] Generating synthetic signal: 3 Hz + 7 Hz sine components (N = " << N << ")...\n";

    std::vector<mathmatrix::fft::Complex> signal(N);
    for (size_t i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / N;
        double val = std::sin(2.0 * 3.14159265 * 3.0 * t) + 0.5 * std::sin(2.0 * 3.14159265 * 7.0 * t);
        signal[i] = mathmatrix::fft::Complex(val, 0.0);
    }

    auto signal_fft = signal;

    // Benchmark O(N log N) FFT
    auto start_fft = std::chrono::high_resolution_clock::now();
    mathmatrix::fft::FastFourierTransform::fft(signal_fft);
    auto end_fft = std::chrono::high_resolution_clock::now();
    auto elapsed_fft_us = std::chrono::duration_cast<std::chrono::microseconds>(end_fft - start_fft).count();

    std::cout << "[+] Radix-2 In-Place FFT Execution Time: " << elapsed_fft_us << " microseconds\n";
    std::cout << "[*] Spectral Peaks Detected in Spectrum:\n";

    for (size_t k = 0; k < 16; ++k) {
        double magnitude = std::abs(signal_fft[k]) / (N / 2);
        if (magnitude > 0.2) {
            std::cout << "    - Frequency Bin " << std::setw(2) << k 
                      << " Hz : Magnitude = " << std::fixed << std::setprecision(4) << magnitude << "\n";
        }
    }

    // Benchmark Matrix Multiplication (64x64)
    std::vector<std::vector<double>> matA(64, std::vector<double>(64, 1.5));
    std::vector<std::vector<double>> matB(64, std::vector<double>(64, 2.0));
    auto res_mat = mathmatrix::fft::FastFourierTransform::matrix_multiply(matA, matB);

    std::cout << "\n[+] 64x64 Matrix Multiplication Completed. Cell [0][0] = " << res_mat[0][0] << "\n";
    std::cout << "[+] MathMatrix C++20 Numerical Lab operational.\n";
    return 0;
}
