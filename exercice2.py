from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Valeurs =[]
if RANK == 0:
    while   True:
        numbre = int(input("entrer un nombre entier postive :"))
        if numbre <0:
            break
else :
    numbre = None    
recvbuf = COMM . bcast ( numbre , root=0 )
#Valeurs.append(recvbuf)
print ("Je suis le process",{RANK} ," J'ai réssu la valeur ",{recvbuf})

    
