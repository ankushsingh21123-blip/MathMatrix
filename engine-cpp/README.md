# MathMatrix C++20 Fast Fourier Transform Engine

High-performance native numerical computing and spectral analysis engine.

## Features
- **In-Place Cooley-Tukey Radix-2 FFT:** $O(N \log N)$ frequency decomposition with bit-reversal indexing and butterfly complex operations.
- **Inverse Fast Fourier Transform (IFFT):** Time-domain waveform reconstruction from frequency spectrums.
- **Matrix Dot Product & Transformations:** Cache-friendly multi-dimensional numeric linear algebra routines.

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
./math_fft_sim
```
