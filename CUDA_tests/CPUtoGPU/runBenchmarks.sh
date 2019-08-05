
#nvcc -I/home/mariog/openmpi-4.0.1/build/include -L/home/mariog/openmpi-4.0.1/build/lib -lmpi bufferSendingBenchmarks.cu -o main
scp -r *  mariog@felk40.cern.ch:/home/mariog/CUDA_tests/CPUtoGPU
#rm ./benchmarksResults/*

#COMMAND="mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n --mca pml ucx --mca btl ^openib main"
#COMMAND="mpirun -np 2 --hostfile hosts --mca btl_openib_allow_ib 1 -x UCX_MEMTYPE_CACHE=n --mca pml ucx --mca btl openib main"
#COMMAND="mpirun -np 2 --hostfile hosts --mca pml ob1 --mca btl openib --mca btl_openib_allow_ib 1 main"
#echo "WITH ETH ----------------------------------------------------------------"
#
#echo "(one step) host to host"
#mpirun -np 2 --hostfile hosts --mca pml ob1 --mca btl ^openib main 1 1 0 0 > ./benchmarksResults/ETH_110.dat
#
#echo "WITHOUT UCX -------------------------------------------------------------"
#
#echo "(one step) host to host"
#mpirun -np 2 --hostfile hosts --mca pml ^ucx --mca btl ^openib main 1 1 0 0 > ./benchmarksResults/noUCX_110.dat
#
#echo "two steps host to device"
#mpirun -np 2 --hostfile hosts --mca pml ^ucx --mca btl ^openib main 0 0 1 0 > ./benchmarksResults/noUCX_001.dat
#

#           # One Step: 1
#           # H to H: 1
#           # H to dev: 0
#           # Only Memcpy: 0



echo "PML UCX; BTL ( ) ----------------------------------------------------"
COMMAND="mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n --mca pml ucx main"
$COMMAND 1 1 0 0 > ./benchmarksResults/ucx_none_110.dat
$COMMAND 1 0 1 0 > ./benchmarksResults/ucx_none_101.dat
#$COMMAND 0 0 1 0 > ./benchmarksResults/ucx_none_001.dat

# ## echo "PML UCX; BTL openIB ----------------------------------------------------"
# ## COMMAND="mpirun -np 2 --hostfile hosts --mca btl_openib_allow_ib 1 -x UCX_MEMTYPE_CACHE=n --mca pml ucx --mca btl openib,self main"
# ## $COMMAND 1 1 0 0 > ./benchmarksResults/ucx_openIB_110.dat
# ## $COMMAND 1 0 1 0 > ./benchmarksResults/ucx_openIB_101.dat
# ## #$COMMAND 0 0 1 0 > ./benchmarksResults/ucx_openIB_001.dat
# ##
# ## echo "PML UCX; BTL vader ----------------------------------------------------"
# ## COMMAND="mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n --mca pml ucx --mca btl vader,self main"
# ## $COMMAND 1 1 0 0 > ./benchmarksResults/UCX_vader_110.dat
# ## $COMMAND 1 0 1 0 > ./benchmarksResults/UCX_vader_101.dat
# ## #$COMMAND 0 0 1 0 > ./benchmarksResults/UCX_vader_001.dat

echo "PML ob1; BTL ^openIB ----------------------------------------------------"
COMMAND="mpirun -np 2 --hostfile hosts --mca pml ob1 --mca btl ^openib --mca btl_openib_allow_ib 1 main"
$COMMAND 1 1 0 0 > ./benchmarksResults/ob1_noOpenIB_110.dat
#$COMMAND 1 0 1 0 > ./benchmarksResults/ob1_noOpenIB_101.dat
#$COMMAND 0 0 1 0 > ./benchmarksResults/ob1_noOpenIB_001.dat

echo "PML ob1; BTL openIB ----------------------------------------------------"
COMMAND="mpirun -np 2 --hostfile hosts --mca pml ob1 --mca btl openib,self --mca btl_openib_allow_ib 1 main"
$COMMAND 1 1 0 0 > ./benchmarksResults/ob1_openIB_110.dat
#$COMMAND 1 0 1 0 > ./benchmarksResults/ob1_openIB_101.dat
#$COMMAND 0 0 1 0 > ./benchmarksResults/ob1_openIB_001.dat

echo "PML ob1; BTL tcp ----------------------------------------------------"
COMMAND="mpirun -np 2 --hostfile hosts --mca pml ob1 --mca btl tcp,self --mca btl_openib_allow_ib 1 main"
$COMMAND 1 1 0 0 > ./benchmarksResults/ob1_tcp_110.dat
#$COMMAND 1 0 1 0 > ./benchmarksResults/ob1_tcp_101.dat
#$COMMAND 0 0 1 0 > ./benchmarksResults/ob1_tcp_001.dat

echo "PML none; BTL none ----------------------------------------------------"
COMMAND="mpirun -np 2 --hostfile hosts --mca btl_openib_allow_ib 1 main"
$COMMAND 1 1 0 0 > ./benchmarksResults/none_none_110.dat
#$COMMAND 1 0 1 0 > ./benchmarksResults/none_none_101.dat
#$COMMAND 0 0 1 0 > ./benchmarksResults/none_none_001.dat



echo "UCX transports ========================================================"
# https://github.com/openucx/ucx/wiki/UCX-environment-parameters

echo "all ----------------------------------------------------"
COMMAND="mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n -x UCX_TLS=all --mca pml ucx --mca btl ^openib main"
$COMMAND 1 1 0 0 > ./benchmarksResults/UCX_transports/UCX_t_all_110.dat
$COMMAND 1 0 1 0 > ./benchmarksResults/UCX_transports/UCX_t_all_101.dat

echo "( ) ----------------------------------------------------"
COMMAND="mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n --mca pml ucx --mca btl ^openib main"
$COMMAND 1 1 0 0 > ./benchmarksResults/UCX_transports/UCX_t_none_110.dat
$COMMAND 1 0 1 0 > ./benchmarksResults/UCX_transports/UCX_t_none_101.dat

#echo "ugni ----------------------------------------------------"
#COMMAND="mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n -x UCX_TLS=ugni --mca pml ucx --mca btl ^openib main"
#$COMMAND 1 1 0 0 > ./benchmarksResults/UCX_transports/UCX_t_ugni_110.dat
#$COMMAND 1 0 1 0 > ./benchmarksResults/UCX_transports/UCX_t_ugni_101.dat

echo "ud ----------------------------------------------------"
COMMAND="mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n -x UCX_TLS=ud --mca pml ucx --mca btl ^openib main"
$COMMAND 1 1 0 0 > ./benchmarksResults/UCX_transports/UCX_t_ud_110.dat
#$COMMAND 1 0 1 0 > ./benchmarksResults/UCX_transports/UCX_t_ud_101.dat

#echo "dc ----------------------------------------------------"
#COMMAND="mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n -x UCX_TLS=dc --mca pml ucx --mca btl ^openib main"
#$COMMAND 1 1 0 0 > ./benchmarksResults/UCX_transports/UCX_t_dc_110.dat
#$COMMAND 1 0 1 0 > ./benchmarksResults/UCX_transports/UCX_t_dc_101.dat

echo "tcp ----------------------------------------------------"
COMMAND="mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n -x UCX_TLS=tcp --mca pml ucx --mca btl ^openib main"
$COMMAND 1 1 0 0 > ./benchmarksResults/UCX_transports/UCX_t_tcp_110.dat
#$COMMAND 1 0 1 0 > ./benchmarksResults/UCX_transports/UCX_t_tcp_101.dat

echo "\rc ----------------------------------------------------"
COMMAND="mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n -x UCX_TLS=\rc --mca pml ucx --mca btl ^openib main"
$COMMAND 1 1 0 0 > ./benchmarksResults/UCX_transports/UCX_t_rc_110.dat
#$COMMAND 1 0 1 0 > ./benchmarksResults/UCX_transports/UCX_t_rc_101.dat

mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n -x UCX_TLS=\rc --mca pml ucx --mca btl ^openib main 1 1 0 0 > ./benchmarksResults/UCX_transports/UCX_t_rc_110.dat