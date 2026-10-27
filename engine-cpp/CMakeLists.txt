cmake_minimum_required(VERSION 3.20)
project(MathMatrixCPPEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(math_fft_sim
    src/main.cpp
    src/fast_fourier.cpp
)

target_compile_options(math_fft_sim PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O3 -pthread>
)
