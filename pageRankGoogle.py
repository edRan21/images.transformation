import numpy as np

# Matriz de enlaces H
H = np.array([
    [0.0, 0.0, 1.0, 0.5], # Enlaces de P1
    [0.5, 0.0, 0.0, 0.0], # Enlaces de P2
    [0.5, 1.0, 0.0, 0.5], # Enlaces de P3
    [0.0, 0.0, 0.0, 0.0]  # Enlaces de P4
    ])

print(f"Matriz de Enlaces:\n {H}\n")

# Número de páginas
n = H.shape[0]

# Factor de damping
d = 0.85

# Matriz de Google
e = np.ones((n, 1)) # Vector de unos
G = d * H + (1-d) * (1/n) * np.dot(e, e.T)

print(f"Matriz de Google:\n {G}\n")

# Vector de inicio
x = np.ones(n)/n

# Método de potencias
tol = 1e-10
max_iter = 100
for i in range(max_iter):
    x_new = np.dot(G, x)
    if np.linalg.norm(x_new - x, ord=1) < tol:
        break
    x = x_new
    
# Resultados
print(f"Vector PageRank final:\n{x}")

# Clasificación de páginas
pages = ['P1', 'P2', 'P3', 'P4']
ranking = sorted(zip(pages, x), key=lambda t: t[1], reverse=True)
print("\nRanking de páginas: ")
for page, score in ranking:
    print(f"{page}: {score:.6f}")