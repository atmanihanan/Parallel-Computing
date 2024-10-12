#########This program compute parallel csc matrix vector multiplication using mpi########

import numpy as np
from scipy.sparse import lil_matrix
from numpy.random import rand, seed
from numba import njit
from mpi4py import MPI

COMM = MPI.COMM_WORLD
nbOfproc = COMM.Get_size()
RANK = COMM.Get_rank()

seed(42)
def matrixVectorMult(A, b, x):
    
    row, col = A.shape
    for i in range(row):
        a = A[i]
        for j in range(col):
            x[i] += a[j] * b[j]

    return 0
########################initialize matrix A and vector b ######################
#matrix sizes
SIZE = 1000

if RANK == 0:
    L = [ SIZE//nbOfproc for i in range(nbOfproc)  ]
    L[nbOfproc-1] = L[nbOfproc-1]  + SIZE%nbOfproc 
    YY = [ np.zeros((i,SIZE ), dtype=np.float64) for i in L]
    Local_size = [[i, SIZE] for i in L]

    # counts = block of each proc
    counts = [i*SIZE for i in L]
    
    ##initialize matrix A and vector b
    # A = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=np.float64)
    # b = np.array([1, 1, 1])
    A = lil_matrix((SIZE, SIZE))
    A[0, :100] = rand(100)
    A[1, 100:200] = A[0, :100]
    A.setdiag(rand(SIZE))
    A = A.toarray()
    b = rand(SIZE)
else :
    A = None
    b = None
    YY = None
    counts = None
#########Send b to all procs and scatter A (each proc has its own local matrix#####
#LocalMatrix = 
matrice_local = COMM . scatter ( YY , root=0 )  
# Scatter the matrix A
b_local = COMM . bcast (b , root=0 ) 
COMM.Scatterv( [A , counts , MPI.DOUBLE],  recvbuf=matrice_local ,root =0)

#####################Compute A*b locally#######################################
LocalX=np.zeros(matrice_local.shape[0])
start = MPI.Wtime()
matrixVectorMult(matrice_local, b_local, LocalX)

stop = MPI.Wtime()
dt= (stop - start)*1000
dt_max= COMM.allreduce(dt, op=MPI.MAX)
if RANK == 0:
    print("CPU time of parallel multiplication is ", dt_max)
##################Gather te results ###########################################
recvbuf = COMM . gather (LocalX , root=0 )  
if RANK ==0:
    X = np.concatenate(recvbuf, axis=0) 
else :
    X=None 
##################Print the results ###########################################
if RANK == 0 :
    X_ = A.dot(b)
    print("The result of A*b using dot is :", np.max(X_ - X))
    #print("The result of A*b using dot is :", X_)
    #print("The result of A*b using parallel version is :", X)
