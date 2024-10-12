import numpy as np 
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

def solve_2d_diff(u, un, nt, dt, dx, dy, nu):
    row, col = u.shape
    # Assigner les conditions initiales
    # Définir une fonction chapeau I.C. : u(0.5<=x<=1 && 0.5<=y<=1) vaut 2
    u[int(0.5 / dy): int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

    # Boucle temporelle
    for n in range(nt + 1):
        un = u.copy()
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                u[i, j] = un[i, j] + nu * dt / dx**2 * (un[i + 1, j] - 2 * un[i, j] + un[i - 1, j]) \
                                      + nu * dt / dy**2 * (un[i, j + 1] - 2 * un[i, j] + un[i, j - 1])

        # Appliquer les conditions aux limites
        u[:, 0] = 1  # Bord gauche
        u[:, -1] = 1  # Bord droit
        u[0, :] = 1  # Bord inférieur
        u[-1, :] = 1  # Bord supérieur

    return u

# Paramètres du problème
nt = 51
nx = 101
ny = 101
nu = 0.05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.25
dt = sigma * dx * dy / nu

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
u = np.ones((ny, nx))  # Créer une matrice de dimensions nx par ny remplie de 1
un = np.ones((ny, nx))  # Matrice temporaire pour les anciennes valeurs de u

# Résoudre l'équation de diffusion
u = solve_2d_diff(u, un, nt, dt, dx, dy, nu)

# Afficher la solution
fig = pyplot.figure(figsize=(7, 5), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u, cmap=cm.viridis)
pyplot.show()
