import numpy as np
from scipy.linalg import solve
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Generowanie macierzy A
def generate_matrix_A(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 1.0
            else:
                A[i, j] = 1.0 / (i + j + 1)
    return A

# Generowanie wektora x
def generate_vector_x(n):
    values = [1, -1] * (n // 2) + ([1] if n % 2 == 1 else [])
    random.shuffle(values)
    return np.array(values, dtype=float)

# Rozwiązywanie układu
def solve_system(A, b):
    return solve(A, b)

# Eksperyment
def experiment(n, precision=np.float64):
    A = generate_matrix_A(n).astype(precision)
    x_true = generate_vector_x(n).astype(precision)
    b = A @ x_true
    x_calc = solve_system(A, b)
    error = np.linalg.norm(x_true - x_calc)
    return error

# Parametry testowe
n_values = [i for i in range(3, 52, 2)]
precisions = [np.float32, np.float64]
precision_labels = ['float32', 'float64']

# Zbieranie wyników
data = []
for n in n_values:
    row = []
    for precision in precisions:
        error = experiment(n, precision)
        row.append(error)
    data.append(row)

# Tworzenie DataFrame: wiersze = rozmiary, kolumny = precyzje
df = pd.DataFrame(data, index=n_values, columns=precision_labels)

# Heatmapa z logarytmiczną skalą kolorów
plt.figure(figsize=(6, 6))
sns.heatmap(
    df,
    annot=True,
    fmt=".1e",
    cmap="YlGnBu",
    norm=plt.matplotlib.colors.LogNorm(),  # logarytmiczna skala
    cbar_kws={'label': 'Błąd (norma 2)'}
)

plt.title("Różnica wartości rzeczywistej i obliczonej")
plt.xlabel("Precyzja")
plt.ylabel("Rozmiar macierzy n")
plt.tight_layout()
plt.show()
