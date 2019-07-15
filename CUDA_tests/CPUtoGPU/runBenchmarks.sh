



echo "WITHOUT UCX -------------------------------------------------------------"

echo "(one step) host to host"
mpirun -np 2 --hostfile hosts --mca pml ^ucx --mca btl ^openib main 1 1 0 > ./benchmarksResults/noUCX_110.dat

echo "one step host to device"
mpirun -np 2 --hostfile hosts --mca pml ^ucx --mca btl ^openib main 1 0 1 > ./benchmarksResults/noUCX_101.dat

echo "two steps host to device"
mpirun -np 2 --hostfile hosts --mca pml ^ucx --mca btl ^openib main 0 0 1 > ./benchmarksResults/noUCX_010.dat

echo "WITH UCX ----------------------------------------------------------------"

echo "(one step) host to host"
mpirun -np 2 --hostfile hosts --mca pml ucx --mca btl ^openib main 1 1 0 > ./benchmarksResults/UCX_110.dat

echo "one step host to device"
mpirun -np 2 --hostfile hosts --mca pml ucx --mca btl ^openib main 1 0 1 > ./benchmarksResults/UCX_101.dat

echo "two steps host to device"
mpirun -np 2 --hostfile hosts --mca pml ucx --mca btl ^openib main 0 0 1 > ./benchmarksResults/UCX_010.dat