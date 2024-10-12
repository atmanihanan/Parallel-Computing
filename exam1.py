from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size ()
RANK = COMM.Get_rank()
if RANK == 0:
    #initialiser la taille de la matrice 
    n = 8

   # déclaration de la matrice 
    matrice= np.zeros((n, n))

    # Initialiser le diagonal de la matrice de 1 a n
    for i in range(n):
        matrice[i, i] = i + 1
    # destribution de la matrice en p matrice avec p c'est le nombre de processuer
    L = [ n// SIZE for i in range(SIZE)  ]
    L[SIZE-1] = L[SIZE-1]  + n%SIZE
    #YY = [ np.zeros((i,i ), dtype=np.float64) for i in L]
    Local_size = [[i, i] for i in L] 
    data = np.split(matrice, np.cumsum(L)[:-1], axis=0)

else:
    data = None

# envoyer les elements de la matrice global 
local_matrice = COMM.scatter(data, root=0)

# Calculer la trace locale
local_trace = np.trace(local_matrice)

# Rassembler les résultats dans le process 0
all_traces = COMM.gather(local_trace, root=0)

# récupération de la trace total 
if RANK == 0:
    total_trace = sum(all_traces)
    print(" la Trace de la matrice est :", total_trace)

