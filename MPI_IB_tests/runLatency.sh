

mpirun -np 2 --hostfile hosts  --mca pml ucx --mca btl '^openib' main > benchmarkResults/latency_0.dat
mpirun -np 2 --host felk40:2  --mca pml ucx --mca btl '^openib' main > benchmarkResults/latencyFtF_0.dat
echo "0 done"
mpirun -np 2 --hostfile hosts  --mca pml ucx --mca btl '^openib' main > benchmarkResults/latency_1.dat
mpirun -np 2 --host felk40:2  --mca pml ucx --mca btl '^openib' main > benchmarkResults/latencyFtF_1.dat
echo "1 done"
mpirun -np 2 --hostfile hosts  --mca pml ucx --mca btl '^openib' main > benchmarkResults/latency_2.dat
mpirun -np 2 --host felk40:2  --mca pml ucx --mca btl '^openib' main > benchmarkResults/latencyFtF_2.dat
echo "2 done"
mpirun -np 2 --hostfile hosts  --mca pml ucx --mca btl '^openib' main > benchmarkResults/latency_3.dat
mpirun -np 2 --host felk40:2  --mca pml ucx --mca btl '^openib' main > benchmarkResults/latencyFtF_3.dat
echo "3 done"
mpirun -np 2 --hostfile hosts  --mca pml ucx --mca btl '^openib' main > benchmarkResults/latency_4.dat
mpirun -np 2 --host felk40:2  --mca pml ucx --mca btl '^openib' main > benchmarkResults/latencyFtF_4.dat 
echo "4 done"
