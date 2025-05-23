import numpy as np
from scipy.linalg import solve, solve_banded
import time
import math

def generate_tridiagonal_matrix_full(n, dtype):
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        if i > 0:
            A[i, i - 1] = -1.0
        A[i, i] = 2.0
        if i < n - 1:
            A[i, i + 1] = -1.0
    return A

def generate_tridiagonal_banded(n, dtype):
    ab = np.zeros((3, n), dtype=dtype)
    ab[0, 1:] = -1.0  # nadprzekątna
    ab[1, :] = 2.0    # przekątna
    ab[2, :-1] = -1.0 # podprzekątna
    return ab

def run_experiment_exact(n_values, dtype):
    results = []
    for n in n_values:
        A = generate_tridiagonal_matrix_full(n, dtype)
        ab = generate_tridiagonal_banded(n, dtype)

        x_true = np.linspace(1, n, n, dtype=dtype)  # dokładne rozwiązanie
        b = A @ x_true  # generujemy b z dokładnym x

        start = time.time()
        x_full = solve(A, b)
        t_full = time.time() - start

        start = time.time()
        x_banded = solve_banded((1, 1), ab, b)
        t_banded = time.time() - start

        # błąd względem rozwiązania "dokładnego"
        error = float(np.linalg.norm(x_banded.astype(np.float64) - x_true.astype(np.float64)) /
                      np.linalg.norm(x_true.astype(np.float64)))

        results.append((n, t_full, t_banded, error))

    return results

def error_digits(err):
    if err == 0:
        return "∞ cyfr zgodnych"
    power = -int(math.floor(math.log10(err)))
    return f"{power} cyfr zgodnych"

def print_table(results, dtype_label):
    print(f"\n=== Wyniki dla {dtype_label} ===")
    print(f"{'n':>6} | {'Czas (full)':>12} | {'Czas (Thomas)':>14} | {'Błąd względny':>15} | {'Cyfry zgodne':>15}")
    print("-" * 80)
    for n, t1, t2, err in results:
        digits = error_digits(err)
        print(f"{n:6d} | {t1:12.6f} | {t2:14.6f} | {err:15.2e} | {digits:>15}")

# Zakres testowanych wartości n
n_values = [10, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000]

# Uruchomienie testów dla float32 i float64
results_32 = run_experiment_exact(n_values, np.float32)
results_64 = run_experiment_exact(n_values, np.float64)

# Wyświetlenie wyników
print_table(results_32, "float32")
print_table(results_64, "float64")
