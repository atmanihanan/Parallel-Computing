from mpi4py import MPI
import random

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
it_max = 5

if SIZE != 2:
    print('This program only works with 2 processes')
    exit()

if RANK == 0:
    #x =  int(input("entrer un nombre entier postive :")) 
    x= 0
    COMM . send ( x , dest=1 )
    #print(f"Iteration 1 : Process {RANK} sending {x} to Process 1")
    
for i in range(it_max):
    if RANK == 1:
        x = COMM.recv ( source=0 )
        print(f"Iteration {i+1}: Process {RANK} received {x} from Process 0")
        x +=1
        COMM . send ( x , dest= 0)
        #print(f"Iteration {i+1}: Process {RANK} sending {x} to Process 0")

    if RANK==0:
        x = COMM.recv(source=1)
        print(f"Iteration {i+1}: Process {RANK} received {x} from Process 1")
        x +=1
        COMM . send ( x , dest=1 )
        #print(f"Iteration {i+1}: Process {RANK} sending {x} to Process 1")
   
