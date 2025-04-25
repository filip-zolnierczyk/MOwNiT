def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

def newton_method(x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Pochodna wynosi zero.")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            print(f"Znaleziono miejsce zerowe po {i+1} iteracjach.")
            return x_new
        x = x_new
    raise Exception("Nie znaleziono miejsca zerowego w podanej liczbie iteracji.")

# Przykład użycia
x0 = 1.5
wynik_newton = newton_method(x0)
print("Metoda Newtona: x =", wynik_newton)
