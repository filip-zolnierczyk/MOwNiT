import numpy as np
from scipy.linalg import solve
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Macierz z punktu 2
def generate_matrix_A2(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            A[i, j] = (2 * (i + 1)) / (j + 1)
    A = A + A.T - np.diag(np.diag(A))
    return A

# Losowy wektor x z permutacji {1, -1}
def generate_vector_x(n):
    values = [1, -1] * (n // 2) + ([1] if n % 2 == 1 else [])
    random.shuffle(values)
    return np.array(values, dtype=float)

# Eksperyment: zwraca błąd i uwarunkowanie
def experiment(n, precision=np.float64):
    A = generate_matrix_A2(n).astype(precision)
    x_true = generate_vector_x(n).astype(precision)
    b = A @ x_true
    x_calc = solve(A, b)
    error = np.linalg.norm(x_true - x_calc)
    cond = np.linalg.cond(A)
    return error, cond

# Parametry
n_values = [i for i in range(10, 1000, 10)]
precisions = [np.float32, np.float64]
precision_labels = ['float32', 'float64']

errors = []
conds = []

for n in n_values:
    row_error = []
    row_cond = []
    for precision in precisions:
        error, cond = experiment(n, precision)
        row_error.append(error)
        row_cond.append(cond)
    errors.append(row_error)
    conds.append(row_cond)

# DataFrames
df_errors = pd.DataFrame(errors, index=n_values, columns=precision_labels)
df_conds = pd.DataFrame(conds, index=n_values, columns=precision_labels)

# Wykres błędów
plt.figure(figsize=(8, 5))
for precision in precision_labels:
    plt.plot(n_values, df_errors[precision], marker='o', label=precision)
plt.yscale("log")
plt.title("Błąd rozwiązania – Macierz z punktu 2")
plt.xlabel("Rozmiar macierzy n")
plt.ylabel("Błąd (||x_true - x_calc||)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Wykres uwarunkowania
plt.figure(figsize=(8, 5))
for precision in precision_labels:
    plt.plot(n_values, df_conds[precision], marker='s', label=precision)
plt.yscale("log")
plt.title("Uwarunkowanie macierzy – Macierz z punktu 2")
plt.xlabel("Rozmiar macierzy n")
plt.ylabel("Uwarunkowanie κ(A)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
