def f(x):
    return x**3 - x - 2

def secant_method(x0, x1, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        f0 = f(x0)
        f1 = f(x1)
        if f1 - f0 == 0:
            raise ZeroDivisionError("Dzielenie przez zero w metodzie siecznych.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            print(f"Znaleziono miejsce zerowe po {i+1} iteracjach.")
            return x2
        x0, x1 = x1, x2
    raise Exception("Nie znaleziono miejsca zerowego w podanej liczbie iteracji.")

# Przykład użycia
x0 = 1.0
x1 = 2.0
wynik_sieczne = secant_method(x0, x1)
print("Metoda siecznych: x =", wynik_sieczne)
