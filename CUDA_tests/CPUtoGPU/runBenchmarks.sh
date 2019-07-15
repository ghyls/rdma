



# WITHOUT UCX ------------------------------------------------------------------

# (one step) host to host
mpirun -np 2 --hostfile hosts --mca pml ^ucx --mca btl ^openib main 1 1 0 > benchmarkResults/noUCX_110

# one step host to device
mpirun -np 2 --hostfile hosts --mca pml ^ucx --mca btl ^openib main 1 0 1 > benchmarkResults/noUCX_101

# two steps host to host
mpirun -np 2 --hostfile hosts --mca pml ^ucx --mca btl ^openib main 0 1 0 > benchmarkResults/noUCX_010



# # WITH UCX ------------------------------------------------------------------
# 
# # (one step) host to host
# mpirun -np 2 --hostfile hosts --mca pml ucx --mca btl ^openib main 1 1 0 > benchmarkResults/UCX_110
# 
# # one step host to device
# mpirun -np 2 --hostfile hosts --mca pml ucx --mca btl ^openib main 1 0 1 > benchmarkResults/UCX_101
# 
# # two steps host to host
# mpirun -np 2 --hostfile hosts --mca pml ucx --mca btl ^openib main 0 1 0 > benchmarkResults/UCX_010