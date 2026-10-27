"""
MathMatrix Scientific Computing Microservice
Provides real-time numerical calculus, Fourier transforms, probability distributions,
and matrix algebra computation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import math
import cmath

app = FastAPI(
    title="MathMatrix Scientific Computing API",
    description="Python backend for numerical analysis, calculus, and Fourier transforms",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FourierRequest(BaseModel):
    frequencies: List[float] = [2.0, 5.0, 9.0]
    sample_rate: int = 128
    duration_seconds: float = 1.0

class CalculusRequest(BaseModel):
    expression_type: str = "polynomial" # polynomial, sin, exp
    lower_bound: float = 0.0
    upper_bound: float = 5.0
    steps: int = 1000

class MatrixRequest(BaseModel):
    matrix_a: List[List[float]]
    matrix_b: List[List[float]]

@app.get("/")
def root():
    return {
        "engine": "MathMatrix Scientific Server",
        "backend": "Python 3.12 FastAPI Service",
        "status": "active"
    }

@app.post("/api/math/fourier")
def compute_discrete_fourier(req: FourierRequest):
    N = int(req.sample_rate * req.duration_seconds)
    time_points = [i / req.sample_rate for i in range(N)]
    
    # Synthesize composite signal
    signal = []
    for t in time_points:
        val = sum(math.sin(2 * math.pi * f * t) for f in req.frequencies)
        signal.append(val)
    
    # Compute Discrete Fourier Transform (DFT)
    dft_magnitudes = []
    frequencies = []
    
    for k in range(N // 2):
        freq = (k * req.sample_rate) / N
        sum_val = complex(0.0, 0.0)
        for n in range(N):
            angle = -2.0 * math.pi * k * n / N
            sum_val += signal[n] * cmath.exp(complex(0, angle))
        
        magnitude = abs(sum_val) / (N / 2)
        dft_magnitudes.append(round(magnitude, 4))
        frequencies.append(round(freq, 2))
        
    return {
        "samples_count": N,
        "input_frequencies": req.frequencies,
        "frequencies": frequencies,
        "magnitudes": dft_magnitudes,
        "detected_peaks": [f for f, m in zip(frequencies, dft_magnitudes) if m > 0.5]
    }

@app.post("/api/math/calculus/integrate")
def numerical_integration(req: CalculusRequest):
    # Function evaluation
    def evaluate(x: float) -> float:
        if req.expression_type == "polynomial":
            return x ** 3 - 2 * (x ** 2) + 4
        elif req.expression_type == "sin":
            return math.sin(x) * x
        elif req.expression_type == "exp":
            return math.exp(-0.5 * (x ** 2))
        return x

    dx = (req.upper_bound - req.lower_bound) / req.steps
    
    # Trapezoidal rule
    area = 0.5 * (evaluate(req.lower_bound) + evaluate(req.upper_bound))
    for i in range(1, req.steps):
        area += evaluate(req.lower_bound + i * dx)
    area *= dx

    # Numerical derivative at midpoint
    midpoint = (req.lower_bound + req.upper_bound) / 2.0
    h = 1e-5
    derivative_midpoint = (evaluate(midpoint + h) - evaluate(midpoint - h)) / (2 * h)

    return {
        "expression": req.expression_type,
        "lower_bound": req.lower_bound,
        "upper_bound": req.upper_bound,
        "definite_integral": round(area, 6),
        "midpoint_sample": round(midpoint, 4),
        "derivative_at_midpoint": round(derivative_midpoint, 6),
        "algorithm": "Trapezoidal Riemann Quadrature & Central Difference"
    }

@app.post("/api/math/matrix/multiply")
def multiply_matrices(req: MatrixRequest):
    A = req.matrix_a
    B = req.matrix_b
    
    rows_A = len(A)
    cols_A = len(A[0]) if rows_A > 0 else 0
    rows_B = len(B)
    cols_B = len(B[0]) if rows_B > 0 else 0
    
    if cols_A != rows_B:
        raise HTTPException(status_code=400, detail=f"Incompatible dimensions: ({rows_A}x{cols_A}) and ({rows_B}x{cols_B})")
    
    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
            
    return {
        "dimensions": f"{rows_A}x{cols_B}",
        "result_matrix": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
