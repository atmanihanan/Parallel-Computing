from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time
COMM= MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
tag =11
# Longueur du domaine 
L = 10
# Nombre de noeuds du maillage
nx = 200  
# Le pas du maillage
dx = L/(nx-1)
# Vitesse du transport
c = 2
# Maillage et condition initiale
x = np.linspace(0,L,nx)
u = np.zeros(nx)
un = np.zeros(nx)
def f(x):
    x = x % L
    if 3<=x<=4:
        return 1
    else:
        return 0
        
for i in range(nx):
    u[i] = f(x[i])
# Tracé de la condition initiale
plt.plot(x,u,'-b')
plt.grid()
# Temps final des simulations
nt=100
# Nombre CFL tel que (0 < CFL <=1)
CFL = 0.8
# Calcul du pas du temps pour assurer la stabilité
dt = CFL*dx/abs(c)

#lamda = a*dt/dx
def solve_1d_linearconv(u, un, nt, nx, dt, dx, c):
    for n in range(nt):  
        for i in range(nx): un[i] = u[i]
        for i in range(1, nx): 
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
    return 0
if RANK ==0:
    # Diviser le vecteur en n sous-vecteurs
    sous_vecteurs = np.array_split(u, SIZE)  
    for i in (1,SIZE):
        # Ajouter une copie du dernier élément de tableau1 au début de tableau2
        dernier_element = sous_vecteurs[i][-1]

# Ajouter le dernier élément copié au début du deuxième tableau
        sous_vecteurs[i+1]=np.insert(sous_vecteurs[i+1],0, dernier_element)
        #sous_vecteurs[i+1]=np.insert(sous_vecteurs[i+1],0, sous_vecteurs[i][-1])
for i in range(SIZE) :
    if RANK==0 :
        COMM.send ( sous_vecteurs[i] , dest=i , tag=tag )
    if RANK == i :
        u_local= COMM.recv ( source=0 , tag=tag )
nx_local=len(u_local)  

un_local= np.zeros(nx)

def solve_1d_linearconv(u_local, un_local, nt, nx, dt, dx, c):
    for n in range(nt):  
        for i in range(nx): un_local[i] = u_local[i]
        for i in range(1, nx): 
            u_local[i] = un_local[i] - c * dt / dx * (un_local[i] - un_local[i-1])
    return 0
if RANK!=0:
    u_local.pop(0)
for i in range(ZISE):
    COMM.send ( u_local , dest=0 )  
    sendbuf= COMM.recv ( source=i )
    tab=[]  
    tab.append(sendbuf)
if RANK==0:
    # Concaténer les éléments des tableaux
    u_parallel = np.concatenate(tab)   
    # Tracer u en fonction de x
    plt.plot(x, u_parallel, marker='o', linestyle='-')

    # Ajouter des titres et des étiquettes
    plt.title('Graphe de u en fonction de x')
    plt.xlabel('x')
    plt.ylabel('u')

    # Afficher le graphe
    plt.grid(True)
    plt.show()
