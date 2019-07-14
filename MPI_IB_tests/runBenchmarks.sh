
# run benchmarks over IB and ETH


echo "0"
mpirun -np 2 --hostfile hosts  --mca pml ucx --mca btl '^openib' main > benchmarkResults/dataIB_0.dat
#mpirun -np 2 --hostfile hosts --mca pml ob1 --mca btl tcp,self main > benchmarkResults/dataTCP_0.dat
mpirun -np 2 --host felk40:2 --mca pml ucx --mca btl ^openib main > benchmarkResults/dataFelkToFelk_0.dat
mpirun -np 2 --hostfile hosts --mca btl_openib_allow_ib 1 --mca btl openib main > benchmarkResults/overIB_0.dat
echo "1"
mpirun -np 2 --hostfile hosts  --mca pml ucx --mca btl '^openib' main > benchmarkResults/dataIB_1.dat
#mpirun -np 2 --hostfile hosts --mca pml ob1 --mca btl tcp,self main > benchmarkResults/dataTCP_1.dat
mpirun -np 2 --host felk40:2 --mca pml ucx --mca btl ^openib main > benchmarkResults/dataFelkToFelk_1.dat 
mpirun -np 2 --hostfile hosts --mca btl_openib_allow_ib 1 --mca btl openib main > benchmarkResults/overIB_1.dat
echo "2"
mpirun -np 2 --hostfile hosts  --mca pml ucx --mca btl '^openib' main > benchmarkResults/dataIB_2.dat
#mpirun -np 2 --hostfile hosts --mca pml ob1 --mca btl tcp,self main > benchmarkResults/dataTCP_2.dat
mpirun -np 2 --host felk40:2 --mca pml ucx --mca btl ^openib main > benchmarkResults/dataFelkToFelk_2.dat 
mpirun -np 2 --hostfile hosts --mca btl_openib_allow_ib 1 --mca btl openib main > benchmarkResults/overIB_2.dat
echo "3"
mpirun -np 2 --hostfile hosts  --mca pml ucx --mca btl '^openib' main > benchmarkResults/dataIB_3.dat
#mpirun -np 2 --hostfile hosts --mca pml ob1 --mca btl tcp,self main > benchmarkResults/dataTCP_3.dat
mpirun -np 2 --host felk40:2 --mca pml ucx --mca btl ^openib main > benchmarkResults/dataFelkToFelk_3.dat 
mpirun -np 2 --hostfile hosts --mca btl_openib_allow_ib 1 --mca btl openib main > benchmarkResults/overIB_3.dat
echo "4"
mpirun -np 2 --hostfile hosts  --mca pml ucx --mca btl '^openib' main > benchmarkResults/dataIB_4.dat
#mpirun -np 2 --hostfile hosts --mca pml ob1 --mca btl tcp,self main > benchmarkResults/dataTCP_4.dat
mpirun -np 2 --host felk40:2 --mca pml ucx --mca btl ^openib main > benchmarkResults/dataFelkToFelk_4.dat 
mpirun -np 2 --hostfile hosts --mca btl_openib_allow_ib 1 --mca btl openib main > benchmarkResults/overIB_4.dat