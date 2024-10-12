
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

def solve_1d_linearconv(u, un, nt, nx, dt, dx, c, comm):
    for n in range(nt):
        un[:] = u[:]
        for i in range(1, nx):
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
        if comm.Get_rank() != 0:
            comm.send(u[1], dest=comm.Get_rank() - 1)
            u[0] = comm.recv(source=comm.Get_rank() - 1)
        if comm.Get_rank() != comm.Get_size() - 1:
            comm.send(u[-2], dest=comm.Get_rank() + 1)
            u[-1] = comm.recv(source=comm.Get_rank() + 1)
    return u

if __name__ == "__main__":
    start_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Simulation parameters
    nx = 200
    nt = 100 
    c = 1
    dx = 2 * np.pi / (nx - 1)
    CFL = 0.8
    dt = CFL * dx / c
    x = np.linspace(0, 2 * np.pi, nx)

    local_nx = nx // size
    remainder = nx % size
    if rank < remainder:
        local_nx += 1

    start_idx = rank * local_nx + min(rank, remainder)
    end_idx = start_idx + local_nx
    if rank < remainder:
        end_idx += 1

    u = np.zeros(local_nx)
    u[:] = np.where((x[start_idx:end_idx] >= 3) & (x[start_idx:end_idx] <= 4), 1, 0)
    un = np.zeros(local_nx)
    print(f'Process {rank}: u = {u}')

    u = solve_1d_linearconv(u, un, nt, local_nx, dt, dx, c, comm)

    # Gather results on root process
    if rank == 0:
        all_results = np.empty(nx)
        all_results[start_idx:end_idx] = u
        for i in range(1, size):
            start_idx_i = i * local_nx + min(i, remainder)
            end_idx_i = start_idx_i + local_nx
            if i < remainder:
                end_idx_i += 1
            u_i = np.empty(local_nx)
            comm.Recv(u_i, source=i)
            all_results[start_idx_i:end_idx_i] = u_i
        end_time = time.time()
        print(f"Process {rank}: {all_results}")
        print("Time taken:", end_time - start_time, "seconds")
        plt.plot(x, all_results)
        plt.show()
        plt.savefig("plot.png")
    else:
        comm.Send(u, dest=0)

