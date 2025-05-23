import numpy as np
from scipy.linalg import solve, solve_banded
import time

# Tworzenie pełnej macierzy A
def generate_tridiagonal_matrix_full(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -2 * (i + 1) - 3  # główna przekątna
        if i < n - 1:
            A[i, i + 1] = i + 1      # nad przekątną
        if i > 0:
            A[i, i - 1] = 2 / (i + 1)  # pod przekątną
    return A

# Wersja zoptymalizowana do solve_banded
def generate_tridiagonal_banded(n):
    ab = np.zeros((3, n))
    ab[0, 1:] = [i for i in range(1, n)]         # nad przekątną
    ab[1, :]  = [-2 * (i + 1) - 3 for i in range(n)]  # główna
    ab[2, :-1] = [2 / (i + 1) for i in range(1, n)]   # pod przekątną
    return ab

# Wektor x i wektor b
def generate_vector_x(n):
    x = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=float)
    return x

# Porównanie metod
def compare_methods(n):
    x_true = generate_vector_x(n)

    # Pełna macierz + solve
    A_full = generate_tridiagonal_matrix_full(n)
    b_full = A_full @ x_true
    start_full = time.time()
    x_full = solve(A_full, b_full)
    end_full = time.time()
    error_full = np.linalg.norm(x_true - x_full)
    time_full = end_full - start_full

    # Trójdiagonalna forma + solve_banded
    ab = generate_tridiagonal_banded(n)
    b_banded = A_full @ x_true  # taki sam wektor b
    start_banded = time.time()
    x_banded = solve_banded((1, 1), ab, b_banded)
    end_banded = time.time()
    error_banded = np.linalg.norm(x_true - x_banded)
    time_banded = end_banded - start_banded

    return {
        "n": n,
        "error_full": error_full,
        "time_full": time_full,
        "error_banded": error_banded,
        "time_banded": time_banded
    }

# Uruchom dla różnych rozmiarów
n_values = [10, 100, 1000, 10000]
results = []

for n in n_values:
    result = compare_methods(n)
    results.append(result)

# Wyświetlenie wyników
print(f"{'n':>6} | {'error_full':>12} | {'time_full (s)':>12} | {'error_banded':>14} | {'time_banded (s)':>14}")
print("-" * 75)
for r in results:
    print(f"{r['n']:>6} | {r['error_full']:>12.2e} | {r['time_full']:>12.4f} | {r['error_banded']:>14.2e} | {r['time_banded']:>14.4f}")
