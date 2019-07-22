
nvcc -I/home/mariog/openmpi-4.0.1/build/include -L/home/mariog/openmpi-4.0.1/build/lib -lmpi bufferSendingBenchmarks.cu -o main
scp -r *  mariog@felk40.cern.ch:/home/mariog/CUDA_tests/CPUtoGPU


echo "WITH ETH ----------------------------------------------------------------"

echo "(one step) host to host"
mpirun -np 2 --hostfile hosts --mca pml ob1 --mca btl ^openib main 1 1 0 0 > ./benchmarksResults/ETH_110.dat

echo "WITHOUT UCX -------------------------------------------------------------"

echo "(one step) host to host"
mpirun -np 2 --hostfile hosts --mca pml ^ucx --mca btl ^openib main 1 1 0 0 > ./benchmarksResults/noUCX_110.dat

echo "two steps host to device"
mpirun -np 2 --hostfile hosts --mca pml ^ucx --mca btl ^openib main 0 0 1 0 > ./benchmarksResults/noUCX_001.dat

echo "WITH UCX ----------------------------------------------------------------"

echo "(one step) host to host"
mpirun -np 2 --hostfile hosts --mca pml ucx --mca btl ^openib main 1 1 0 0 > ./benchmarksResults/UCX_110.dat

echo "one step host to device"
mpirun -np 2 --hostfile hosts --mca pml ucx --mca btl ^openib main 1 0 1 0 > ./benchmarksResults/UCX_101.dat

echo "two steps host to device"
mpirun -np 2 --hostfile hosts --mca pml ucx --mca btl ^openib main 0 0 1 0 > ./benchmarksResults/UCX_001.dat

echo "PATATRACK cudaMemcpy() --------------------------------------------------"
mpirun -np 2 --hostfile hosts --mca pml ucx --mca btl ^openib main 0 0 0 1 > ./benchmarksResults/Memcpy.dat
