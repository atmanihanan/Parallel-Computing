from mpi4py import MPI
import numpy as np

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

def distribute_data(matrice, size):
    n = matrice.shape[0]
    part_size = n // size
    reste = n % size

    data = []
    start_row = 0
    for i in range(size):
        end_row = start_row + part_size + (1 if i < reste else 0)
        data.append(matrice[start_row:end_row, :])
        start_row = end_row

    return data

if RANK == 0:
    # initialiser la taille de la matrice
    n = 8

    # déclaration de la matrice
    matrice = np.zeros((n, n))

    # Initialiser la diagonale de la matrice de 1 à n
    for i in range(n):
        matrice[i, i] = i + 1

    # Distribuer la matrice en parties 
    data = distribute_data(matrice, SIZE)
else:
    data = None

# Envoyer les parties de la matrice aux autres processus (les sous matrice)
local_matrice = COMM.scatter(data, root=0)

# Calculer la somme locale des éléments
local_sum = np.sum(local_matrice)

# Rassembler les résultats dans le processus 0
all_sums = COMM.gather(local_sum, root=0)

# Calculer la trace globale
if RANK == 0:
    global_sum = sum(all_sums)
    print("La trace de la matrice est  :", global_sum)


