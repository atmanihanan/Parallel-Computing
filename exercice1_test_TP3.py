import random
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
nb_procs = comm.Get_size()
rank = comm.Get_rank()
from utils import compute_dims

def creer_matrice(n, m):
    # Créer une matrice de dimensions n x m avec des valeurs aléatoires entre 0 et 1
    matrice = np.random.randint(2, size=(n, m))
    return matrice

n = 4
m = 4
M = creer_matrice(n, m)
# periods = tuple([False, False])
reorder = False
dims, blockSize = compute_dims(nb_procs, [n, m], min_blocksizes=None, mpi=None)
cart = comm.Create_cart(dims=dims, periods=None, reorder=reorder)
coord = cart.Get_coords(rank)
start_x = coord[0] * blockSize[0]
end_x = coord[0] * blockSize[0] + blockSize[0]
start_y = coord[1] * blockSize[1]
end_y = coord[1] * blockSize[1] + blockSize[1]
local_M = np.zeros((blockSize[0] + 2, blockSize[1] + 2))
local_M[1:-1, 1:-1] = M[start_x:end_x, start_y:end_y]

# get direction
left, right = cart.Shift(direction=1, disp=1)
high, low = cart.Shift(direction=0, disp=1)
nb_tronsition = 4

fig, ax = plt.subplots()  # Créer une seule fois la figure
im = ax.imshow(M, cmap='binary')

plt.show(block=False)  # Afficher la figure une seule fois avant la boucle

for transition in range(nb_tronsition):
    # if right>=0:
    sendbuf = local_M[:, -2]
    col_left = cart.sendrecv(sendbuf, dest=right, source=left)
    # if left>=0:
    sendbuf = local_M[:, 1]
    col_right = cart.sendrecv(sendbuf, dest=left, source=right)
    # if low>=0:
    sendbuf = local_M[-2, :]
    ligne_high = cart.sendrecv(sendbuf, dest=low, source=high)
    # if high>=0:
    sendbuf = local_M[1, :]
    ligne_low = cart.sendrecv(sendbuf, dest=high, source=low)

    # local_M[:, 0]=col_left
    # local_M[:, -1]=col_right
    local_M[0, :] = ligne_high
    local_M[-1, :] = ligne_low

    nombre_lignes = len(local_M)
    nombre_colonnes = len(local_M[0])

    # Copie de la matrice
    local_M_copie = np.copy(local_M)
    for i in range(1, nombre_lignes - 1):
        for j in range(1, nombre_colonnes - 1):
            if local_M_copie[i][j] == 1:
                Voisinage_Vivre = -1
                for k in range(i - 1, i + 2):
                    for h in range(j - 1, j + 2):
                        if local_M_copie[k][h] == 1:
                            Voisinage_Vivre += 1
                if Voisinage_Vivre == 1 or Voisinage_Vivre == 0:
                    local_M[i][j] = 0
                elif Voisinage_Vivre >= 4:
                    local_M[i][j] = 0
            else:
                Voisinage_Vivre = 0
                for k in range(i - 1, i + 2):
                    for h in range(j - 1, j + 2):
                        if local_M_copie[k][h] == 1:
                            Voisinage_Vivre += 1
                if Voisinage_Vivre == 3:
                    local_M[i][j] = 1

    recvbuf = comm.gather(local_M, root=0)
    if rank == 0:
        for i in range(nb_procs):
            block_matrice = recvbuf[i][1:-1, 1:-1]
            coord = cart.Get_coords(i)
            start_x = coord[0] * blockSize[0]
            end_x = coord[0] * blockSize[0] + blockSize[0]
            start_y = coord[1] * blockSize[1]
            end_y = coord[1] * blockSize[1] + blockSize[1]
            M[start_x:end_x, start_y:end_y] = block_matrice

        # print(recvbuf)
        # print([cart.Get_coords(i) for i in range(nb_procs)])
        # print(blockSize)
        print(f"Transition : {transition}")
        print(M)

        im.set_data(M)  # Mettre à jour l'image avec les nouvelles données
        plt.draw()
        plt.pause(0.1)

plt.ioff()  # Désactiver le mode interactif pour éviter l'affichage de nouvelles fenêtres
plt.show()
