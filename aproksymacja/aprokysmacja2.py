import numpy as np
import matplotlib.pyplot as plt


# Funkcja oryginalna
def f(x):
    return x * np.sin(2 * np.pi / x)


# Parametry
a, b = 0.125, 0.4
n_values = [20, 50, 100]
poly_degrees = [3, 5, 10]
trig_terms_list = [3, 5, 10]


# Aproksymacja wielomianowa (ręcznie)
def polynomial_approx(x, y, degree):
    A = np.vander(x, N=degree + 1, increasing=True)
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    y_approx = A @ coeffs
    return y_approx


# Aproksymacja trygonometryczna (ręcznie)
def trigonometric_approx(x, y, k_max):
    omega = np.pi / (b - a)
    A = [np.ones_like(x)]
    for k in range(1, k_max + 1):
        A.append(np.cos(k * omega * x))
        A.append(np.sin(k * omega * x))
    A = np.vstack(A).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    y_approx = A @ coeffs
    return y_approx


# Funkcje błędów
def error_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    max_err = np.max(np.abs(y_true - y_pred))
    return rmse, max_err


# Wyniki
print(f"{'n':>4} | {'deg/k':>6} | {'Typ':^15} | {'RMSE':>10} | {'MaxErr':>10}")
print("-" * 55)

for n in n_values:
    x = np.linspace(a, b, n)
    y = f(x)

    # Wielomiany
    for deg in poly_degrees:
        y_poly = polynomial_approx(x, y, deg)
        rmse, max_err = error_metrics(y, y_poly)
        print(f"{n:4d} | {deg:6d} | {'wielomian':^15} | {rmse:10.3e} | {max_err:10.3e}")

    # Trygonometryczne
    for k in trig_terms_list:
        y_trig = trigonometric_approx(x, y, k)
        rmse, max_err = error_metrics(y, y_trig)
        print(f"{n:4d} | {k:6d} | {'trygonometryczna':^15} | {rmse:10.3e} | {max_err:10.3e}")
