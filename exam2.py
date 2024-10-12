from mpi4py import MPI
import numpy as np

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

if RANK == 0:
    # initialiser la taille de la matrice
    n = 8

    # déclaration de la matrice
    matrice = np.zeros((n, n))

    # Initialiser la diagonale de la matrice de 1 à n
    for i in range(n):
        matrice[i, i] = i + 1

    # Distribuer la matrice en p parties avec p étant le nombre de processus
    L = [n // SIZE for i in range(SIZE)]
    L[SIZE - 1] += n % SIZE
    Local_size = [[i, i] for i in L]
    
    # Diviser la matrice en parties
    data = np.split(matrice, np.cumsum(L)[:-1], axis=0)

else:
    data = None

# Envoyer les parties de la matrice aux autres processus
local_matrice = COMM.scatter(data, root=0)

# Calculer la trace locale
local_trace = np.trace(local_matrice)

# Rassembler les résultats dans le processus 0
all_traces = COMM.gather(local_trace, root=0)

# récupération de la trace totale
if RANK == 0:
    total_trace = sum(all_traces)
    print("la Trace de la matrice est :", total_trace)
