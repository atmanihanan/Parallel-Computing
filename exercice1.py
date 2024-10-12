
import random
from mpi4py import MPI

def compute_points(INTERVAL):
    circle_points = 0
    for _ in range(INTERVAL**2):
        rand_x = random.uniform(-1, 1)
        rand_y = random.uniform(-1, 1)
        origin_dist = rand_x**2 + rand_y**2
        if origin_dist <= 1:
            circle_points += 1
    return circle_points

def monte_carlo_parallel(N):
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size ()
    RANK = COMM.Get_rank()
    sendbuf = None
    if RANK == 0:
        sendbuf= [N//SIZE for i in range(SIZE)  ]
        sendbuf[SIZE-1] = sendbuf[SIZE-1]  + N%SIZE 
        summ = sum([i**2 for i in sendbuf])
    recvbuf = COMM.scatter(sendbuf , root=0)

    nem_point=compute_points(recvbuf)      
    sum_reduce = COMM.reduce( nem_point , op=MPI.SUM , root = 0)
    if RANK == 0:
        pi= 4*sum_reduce/summ
        return pi
print(monte_carlo_parallel(10000))   
