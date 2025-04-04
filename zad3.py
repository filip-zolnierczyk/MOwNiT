import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')
plt.ion()


def f(x):
    return x * np.sin((2 * np.pi) / x)


def df(x):
    return np.sin((2 * np.pi) / x) + x * (2 * np.pi / x ** 2) * np.cos((2 * np.pi) / x)


def chebyshev_nodes(a, b, n):
    return np.array([(a + b) / 2 + (b - a) / 2 * np.cos((2 * i + 1) / (2 * n) * np.pi) for i in range(n)])


def hermite_interpolation(x_nodes, y_nodes, dy_nodes, x):
    n = len(x_nodes)
    Q = np.zeros((2 * n, 2 * n))
    Z = np.zeros(2 * n)

    for i in range(n):
        Z[2 * i] = Z[2 * i + 1] = x_nodes[i]
        Q[2 * i, 0] = Q[2 * i + 1, 0] = y_nodes[i]
        Q[2 * i + 1, 1] = dy_nodes[i]

        if i != 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (Z[2 * i] - Z[2 * i - 1])

    for j in range(2, 2 * n):
        for i in range(j, 2 * n):
            Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (Z[i] - Z[i - j])

    result = Q[-1, -1]
    for i in range(2 * n - 2, -1, -1):
        result = result * (x - Z[i]) + Q[i, i]

    return result


def interpolate_and_analyze(f, df, a, b, degrees, method='chebyshev', num_test_points=1000):
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
        dy_nodes = df(x_nodes)
        interpolator_hermite = np.array([hermite_interpolation(x_nodes, y_nodes, dy_nodes, x) for x in x_test])

        error_max_hermite = np.max(np.abs(interpolator_hermite - y_true))
        error_mean_hermite = np.mean(np.abs(interpolator_hermite - y_true))
        errors[n] = (error_mean_hermite, error_max_hermite)

        plt.plot(x_test, interpolator_hermite, label=f'Interpolacja Hermite (n={n})')
        plt.scatter(x_nodes, f(x_nodes), marker='o', label=f'Węzły (n={n})')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolacja Hermite’a dla różnych stopni wielomianu')
    plt.savefig("wykres.png")

    print("Stopień | Błąd średni (Hermite)    | Błąd max (Hermite)")
    print("-----------------------------------------------------")
    for n, (err_mean, err_max) in errors.items():
        print(f"{n:7} | {err_mean:.15f} | {err_max:.15f}")


# Przykład użycia
interpolate_and_analyze(f, df, 0.125, 0.4, degrees=[25], method='uniform')#chebyshev
