import numpy as np
from mpmath import mp

# Ustawienie precyzji dla mpmath
mp.dps = 36

def f(x):
    return mp.sqrt(x ** 2 + 1) - 1

def g(x):
    return x ** 2 / (mp.sqrt(x ** 2 + 1) + 1)

n = int(input("Podaj liczbę: "))
if n <= 1:
    print("Podaj większą liczbę")
else:
    exponents = np.arange(1, n + 1)
    x_values = [mp.mpf(8) ** -e for e in exponents]

    types = {
        "float32": np.float32,
        "float64 (double)": np.float64,
        "mpmath (long double)": mp.mpf
    }

    table_format = "{:<6} {:<20} {:<20} {:<20} {:<20}"

    for type_name, dtype in types.items():
        print(f"\nTyp: {type_name}")
        print(table_format.format("n+1", "x", "f(x)", "g(x)", "Różnica"))
        print("-" * 90)

        for i, x in enumerate(x_values, start=1):  # Dodanie numeracji od 1
            x_typed = dtype(x)
            f_val = f(x_typed)
            g_val = g(x_typed)
            diff = abs(f_val - g_val)

            print(table_format.format(
                i,
                f"{float(x):.10e}",
                f"{float(f_val):.10e}",
                f"{float(g_val):.10e}",
                f"{float(diff):.10e}"))
