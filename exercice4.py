from mpi4py import MPI
import random

COMM= MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
if RANK==0:
    data = 10
    COMM.send(data, dest=1)  
    
if 0 <  RANK <SIZE:
    data= COMM.recv(source = RANK -1)
    print(f" Process {RANK} received {data} from Process {RANK-1}")
    data +=RANK
    if RANK!= SIZE -1:
        COMM.send(data,  dest= RANK+1)   

    
