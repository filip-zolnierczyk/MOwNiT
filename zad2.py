import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')
plt.ion()

def f(x):
    return x * np.sin((2 * np.pi) / x)


def chebyshev_nodes(a, b, n):
    return np.array([(a + b) / 2 + (b - a) / 2 * np.cos((2 * i + 1) / (2 * n) * np.pi) for i in range(n)])


def lagrange_interpolation(x_nodes, y_nodes, x):
    n = len(x_nodes)
    result = 0
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if i != j:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result


def divided_differences(x_nodes, y_nodes):
    n = len(x_nodes)
    coef = np.copy(y_nodes)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x_nodes[i] - x_nodes[i - j])
    return coef


def newton_interpolation(x_nodes, y_nodes, x):
    coef = divided_differences(x_nodes, y_nodes)
    n = len(x_nodes)
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x - x_nodes[i]) + coef[i]
    return result


def find_difference_position(num1, num2, precision=15):
    str1 = f"{num1:.{precision}f}"
    str2 = f"{num2:.{precision}f}"

    for i in range(len(str1)):
        if str1[i] != str2[i]:
            return i - str1.index('.')  # Pozycja po przecinku
    return None


def interpolate_and_analyze(f, a, b, degrees, method='chebyshev', num_test_points=1000):
    x_test = np.linspace(a, b, num_test_points)
    y_true = f(x_test)

    errors = {}

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_true, 'k-', label='Funkcja rzeczywista')

    for n in degrees:
        if method == 'uniform':
            x_nodes = np.linspace(a, b, n)
        elif method == 'chebyshev':
            x_nodes = chebyshev_nodes(a, b, n)
        else:
            raise ValueError("Nieznana metoda wyboru węzłów")

        y_nodes = f(x_nodes)
        interpolator_lagrange = np.array([lagrange_interpolation(x_nodes, y_nodes, x) for x in x_test])
        interpolator_newton = np.array([newton_interpolation(x_nodes, y_nodes, x) for x in x_test])

        error_max_lagrange = np.max(np.abs(interpolator_lagrange - y_true))
        error_mean_lagrange = np.mean(np.abs(interpolator_lagrange - y_true))

        error_max_newton = np.max(np.abs(interpolator_newton - y_true))
        error_mean_newton = np.mean(np.abs(interpolator_newton - y_true))

        diff_pos = find_difference_position(error_max_lagrange, error_max_newton, precision=15)

        errors[n] = ((error_mean_lagrange, error_max_lagrange),
                     (error_mean_newton, error_max_newton), diff_pos)

        plt.plot(x_test, interpolator_lagrange, label=f'Interpolacja Lagrange (n={n})')
        plt.plot(x_test, interpolator_newton, label=f'Interpolacja Newton (n={n})')
        plt.scatter(x_nodes, f(x_nodes), marker='o', label=f'Węzły (n={n})')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolacja dla różnych stopni wielomianu')
    #plt.savefig("wykres.png")
    print(
        "Stopień | Błąd średni (Lagrange)   | Błąd max (Lagrange)   | Błąd średni (Newton)   | Błąd max (Newton)   | Różnica na pozycji")
    print(
        "----------------------------------------------------------------------------------------------------------------------")

    for n, ((err_mean_lag, err_max_lag), (err_mean_new, err_max_new), diff_pos) in errors.items():
        diff_text = diff_pos if diff_pos is not None else "Brak różnicy"
        print(
            f"{n:7} | {err_mean_lag:.15f} | {err_max_lag:.15f} | {err_mean_new:.15f} | {err_max_new:.15f} | {diff_text}")


# Przykład użycia
interpolate_and_analyze(f, 0.125, 0.4, degrees=[i for i in range(2, 75)], method='uniform') #uniform, chebyshev
