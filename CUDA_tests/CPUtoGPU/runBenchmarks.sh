



# WITHOUT UCX

mpirun -np 2 --hostfile hosts --mca pml ucx --mca btl ^openib main
