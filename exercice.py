from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

while True:
    if RANK == 0:
        number = int(input('Enter a number (negative integer to stop): '))
        if number < 0:
            for i in range(1, SIZE):
                COMM.send(number, dest=i, tag=1)
            break
        for i in range(1, SIZE):
            COMM.send(number, dest=i, tag=1)
    else: 
        number = COMM.recv(source=0, tag=1)
        if number < 0:
            break
        print(f'Process {RANK} got {number}')
print(f'Process {RANK} stopped')
