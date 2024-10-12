#exercice2
import random
from mpi4py import MPI
import numpy as np
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size ()
RANK = COMM.Get_rank()
N = 1000
x = [np.random.uniform(0,1) for i in range(N)]
y = [2*x + np.random.normal(0, 1) for x in x]

def compute_gradiant(x,y,w,b):
    N=len(x)
    dJ_w = (2/N)*(sum([ (w*x[i] +b -y[i])*x[i] for i in range(N)]))
    dJ_b = (2/N)*(sum([ w*x[i] +b -y[i] for i in range(N)]))
    return dJ_w, dJ_b

## calcule parallél
def  gradient_stochastique(w_0, b_0,x,y):
    h = 0.01
    eps = 0.0001
    w = w_0
    b = b_0
    itr_max= 100
    while np.linalg.norm(compute_gradiant(x,y,w_0,b_0))>eps:
        w = w - h*compute_gradiant(x,y,w,b)[0]
        b = b - h*compute_gradiant(x, y,w,b)[1]
        if itr_max >= 100:
            break
    return w ,b 

def calcul_parallel(x, y, w, b) :
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size ()
    RANK = COMM.Get_rank()
    L = None
    sendbuf_x= []
    sendbuf_y= []
    if RANK == 0:
        #sendbuf_w= []
        #sendbuf_b= []
        L = [ N//SIZE for i in range(SIZE)  ]
        L[SIZE-1] = L[SIZE-1]  + N%SIZE 
        for ind,i in enumerate(L):
            s = sum(L[:ind])
            print(s, s+i)
            x_local = [x[j] for j in range(s, i+s, 1)]
            y_local = [y[j] for j in range(s, i+s, 1)]

            sendbuf_x.append(x_local)
            sendbuf_y.append(y_local)
            #sendbuf_w.append(w_local)
            #sendbuf_b.append(b_local)    
                
    recvbuf_x = COMM.scatter(sendbuf_x , root=0)
    recvbuf_y = COMM.scatter(sendbuf_y , root=0)
    #recvbuf_w = COMM.scatter(sendbuf_w , root=0)
    #recvbuf_b = COMM.scatter(sendbuf_b , root=0)

    # iteration
    h = 0.01
    eps = 0.0001
    w = 1
    b = 0
    itr_max= 1000
    i=0
    while True:
        grad_local = compute_gradiant(recvbuf_x,recvbuf_y,w,b)
        COMM.Barrier()
        sum_reduce_dw = COMM.allreduce(grad_local[0], op=MPI.SUM)
        sum_reduce_db = COMM.allreduce(grad_local[1], op=MPI.SUM)
        w = w - h*sum_reduce_dw
        b = b - h*sum_reduce_db
        i += 1

        if np.linalg.norm([sum_reduce_dw, sum_reduce_db])<eps or i >= itr_max:
            break
    
    if RANK == 0:
        print(f"iter: {i}")
        return [w ,b]

    """
    grad_local_w, grad_local_b = compute_gradiant(recvbuf_x,recvbuf_y,w,b)
       
      
    sum_reduce_dw = COMM.allreduce( grad_local_w, op=MPI.SUM , root = 0)
    sum_reduce_db = COMM.allreduce(  grad_local_b, op=MPI.SUM , root = 0)
   
    if RANK == 0:
        return [sum_reduce_dw, sum_reduce_db]"""

RESULT = calcul_parallel(x, y, 1, 0)
print(f"[w_optimal,b_optimal]={RESULT}") 

"""
def  gradient_stochastique_parallel(x,y,w_0,b_0):
    h = 0.01
    eps = 0.0001
    w = w_0
    b = b_0
    itr_max= 100
    i=0
    while True:
        gradient = calcul_parallel(x,y,w,b)
        print("gradient: ")
        print(gradient)
        w = w - h*gradient[0]
        b = b - h*gradient[1]
        i += 1
        if np.linalg.norm(gradient)>eps or i >= itr_max:
            break
    return w ,b 

if RANK ==0:
    resutat = gradient_stochastique_parallel(x,y, 1,0)
    print(resutat)
"""
