import numpy as np
import matplotlib.pyplot as plt

# Funkcja oryginalna
def f(x):
    return x * np.sin(2 * np.pi / x)

# Parametry
a, b = 0.125, 0.4
n_points = 100 # punkty dyskretyzacji
poly_degree = 5# stopień wielomianu
trig_terms = 5  # pary (sin, cos) do k=5

# Dyskretyzacja
x = np.linspace(a, b, n_points)
y = f(x)

# ----------------------
# Aproksymacja WIELOMIANOWA
# ----------------------
def polynomial_approx(x, y, degree):
    A = np.vander(x, N=degree+1, increasing=True)  # macierz bazowa [1, x, x^2, ..., x^d]
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]   # współczynniki a_i
    y_approx = A @ coeffs
    return y_approx, coeffs

# ----------------------
# Aproksymacja TRYGNOMETRYCZNA
# ----------------------
def trigonometric_approx(x, y, k_max):
    n = len(x)
    omega = np.pi / (b - a)  # podstawowa częstość
    A = [np.ones_like(x)]
    for k in range(1, k_max + 1):
        A.append(np.cos(k * omega * x))
        A.append(np.sin(k * omega * x))
    A = np.vstack(A).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    y_approx = A @ coeffs
    return y_approx, coeffs

# Obliczenia
y_poly, coeffs_poly = polynomial_approx(x, y, poly_degree)
y_trig, coeffs_trig = trigonometric_approx(x, y, trig_terms)

# Błędy
err_poly = np.sqrt(np.mean((y - y_poly)**2))
err_trig = np.sqrt(np.mean((y - y_trig)**2))
print(f"Błąd średniokwadratowy (wielomian ręcznie): {err_poly:.5e}")
print(f"Błąd średniokwadratowy (trygonometryczna ręcznie): {err_trig:.5e}")

# Wykres
plt.figure(figsize=(10,6))
plt.plot(x, y, label='f(x)', linewidth=2)
plt.plot(x, y_poly, '--', label=f'Wielomian stopnia {poly_degree}')
#plt.plot(x, y_trig, ':', label=f'Trygonometryczna ({2*trig_terms+1} składników)')
plt.legend()
plt.grid(True)
plt.title('Aproksymacja funkcji f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
