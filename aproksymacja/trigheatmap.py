import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Funkcja oryginalna
def f(x):
    return x * np.sin(2 * np.pi / x)

# Parametry przedziału
a, b = 0.125, 0.4

# Transformacja przedziału [a, b] → [−π, π]
def scale_to_pi(x):
    return 2 * np.pi * (x - a) / (b - a) - np.pi

# Aproksymacja trygonometryczna
def trigonometric_approx(x_scaled, y, k_max):
    A = [np.ones_like(x_scaled)]
    for k in range(1, k_max + 1):
        A.append(np.cos(k * x_scaled))
        A.append(np.sin(k * x_scaled))
    A = np.vstack(A).T
    try:
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
        y_approx = A @ coeffs
        return np.mean((y - y_approx) ** 2)
    except np.linalg.LinAlgError:
        return np.nan

# Zakres wartości k i liczby węzłów n
k_values = np.arange(1, 101)         # k od 1 do 200
n_values = np.arange(20, 101)   # liczba węzłów od 20 do 200 co 10

# Macierz MSE
mse_matrix = np.zeros((len(n_values), len(k_values)))

for i, n in enumerate(n_values):
    x_nodes = np.linspace(a, b, n)
    x_scaled = scale_to_pi(x_nodes)
    y_nodes = f(x_nodes)
    for j, k in enumerate(k_values):
        mse_matrix[i, j] = trigonometric_approx(x_scaled, y_nodes, k)

# Tworzenie heatmapy
plt.figure(figsize=(12, 6))
im = plt.imshow(mse_matrix, aspect='auto', origin='lower', cmap='viridis',
                extent=[k_values[0], k_values[-1], n_values[0], n_values[-1]],
                norm=LogNorm(vmax=np.nanmax(mse_matrix), vmin=np.nanmin(mse_matrix[np.nonzero(mse_matrix)]))
)

plt.colorbar(im, label="MSE (Błąd średniokwadratowy)")
plt.xlabel("Liczba składników $k$")
plt.ylabel("Liczba węzłów $n$")
plt.title("Heatmapa błędu MSE aproksymacji funkcji $x \sin(2\pi/x)$")
plt.tight_layout()
plt.show()
