import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Funkcja oryginalna
def f(x):
    return x * np.sin(2 * np.pi / x)

# Parametry
a, b = 0.125, 0.4
max_nodes = 250

# Aproksymacja wielomianowa + ewaluacja na x_eval
def polynomial_approx(x, y, degree, x_eval):
    A = np.vander(x, N=degree + 1, increasing=True)
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    A_eval = np.vander(x_eval, N=degree + 1, increasing=True)
    y_eval = A_eval @ coeffs
    return y_eval

# Zakresy dla liczby węzłów i stopni wielomianów
node_values = list(range(3, max_nodes + 1, 2))
degree_values = list(range(3, max_nodes))
error_matrix = np.full((len(node_values), len(degree_values)), np.nan)

# Gęsta siatka testowa
x_test = np.linspace(a, b, 1000)
y_test = f(x_test)

# Liczenie MSE (Mean Squared Error - Błąd średniokwadratowy)
for i, n_nodes in enumerate(node_values):
    x_nodes = np.linspace(a, b, n_nodes)
    y_nodes = f(x_nodes)

    for j, degree in enumerate(degree_values):
        if degree >= n_nodes:
            continue

        y_approx = polynomial_approx(x_nodes, y_nodes, degree, x_test)
        mse = np.mean((y_test - y_approx) ** 2)  # MSE zamiast RMSE
        error_matrix[i, j] = mse

# Wizualizacja
plt.figure(figsize=(14, 9))
masked_matrix = np.ma.masked_invalid(error_matrix)

vmin = np.nanmin(error_matrix)
vmax = np.nanmax(error_matrix)
print(f"Zakres błędów: min={vmin:.2e}, max={vmax:.2e}")

plt.imshow(masked_matrix, origin='lower', cmap='YlGnBu_r',
           norm=LogNorm(vmin=vmin, vmax=vmax),
           extent=[min(degree_values), max(degree_values),
                   min(node_values), max(node_values)],
           aspect='auto')

plt.colorbar(label="MSE (Błąd średniokwadratowy)")
plt.xlabel("Stopień wielomianu")
plt.ylabel("Liczba węzłów")
plt.title("Błąd MSE aproksymacji wielomianowej")
plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()