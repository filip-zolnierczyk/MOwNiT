import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

# Punkt 1: Macierz A1
def generate_matrix_A1(n):
    A = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = 1 / (i + j + 1)
    return A

# Punkt 2: Macierz A2 (jak wcześniej)
def generate_matrix_A2(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            A[i, j] = (2 * (i + 1)) / (j + 1)
    A = A + A.T - np.diag(np.diag(A))
    return A

# Oblicz uwarunkowanie dla każdej macierzy
n_values = list(range(10, 1001, 10))  # możesz rozszerzyć np. do 1000
conds_A1 = []
conds_A2 = []

for n in n_values:
    A1 = generate_matrix_A1(n)
    A2 = generate_matrix_A2(n)
    conds_A1.append(np.linalg.cond(A1))
    conds_A2.append(np.linalg.cond(A2))

# Wykres porównujący uwarunkowanie
plt.figure(figsize=(10, 6))
plt.plot(n_values, conds_A1, label="Macierz z punktu 1", marker='o')
plt.plot(n_values, conds_A2, label="Macierz z punktu 2", marker='s')
plt.yscale("log")
plt.title("Porównanie uwarunkowania macierzy (punkt 1 vs punkt 2)")
plt.xlabel("Rozmiar macierzy n")
plt.ylabel("Uwarunkowanie κ(A)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.tight_layout()
plt.show()
