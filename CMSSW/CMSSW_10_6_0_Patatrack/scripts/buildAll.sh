#!/bin/bash
set -euxo pipefail

echo "======================"
echo "REMEMBER TO RUN cmsenv"
echo "======================"

rm -rf cd $CMSSW_BASE/build/*
rm -rf cd $CMSSW_BASE/local/*


export ZLIB_BASE=$(scram tool tag zlib ZLIB_BASE)

export CUDA_BASE=$(scram tool tag cuda CUDA_BASE)
export LD_LIBRARY_PATH=$CUDA_BASE/drivers:$LD_LIBRARY_PATH

# numactl (20190717)
# ==================

## exported environment
export NUMACTL_DATE=20190717
export NUMACTL_HASH=29d53d8a91fee50aab47240693721a0376091daa
export NUMACTL_BASE=$CMSSW_BASE/local/numactl/$NUMACTL_DATE
export PATH=$NUMACTL_BASE/bin:$PATH
export LD_LIBRARY_PATH=$NUMACTL_BASE/lib:$LD_LIBRARY_PATH

## download, configure, build and install
cd $CMSSW_BASE
mkdir -p build && cd build
git clone https://github.com/numactl/numactl.git numactl-$NUMACTL_DATE
cd numactl-$NUMACTL_DATE
git reset --hard $NUMACTL_HASH

./autogen.sh
mkdir build
cd build
../configure --with-pic --prefix=$NUMACTL_BASE

make -j3
make install

# libnl 3.2.25
# ============

## exported environment
export LIBNL_BASE=$CMSSW_BASE/local/libnl/3.2.25
export LD_LIBRARY_PATH=$LIBNL_BASE/lib:$LD_LIBRARY_PATH

## download, configure, build and install
cd $CMSSW_BASE/build

wget https://www.infradead.org/~tgr/libnl/files/libnl-3.2.25.tar.gz
tar xaf libnl-3.2.25.tar.gz
cd libnl-3.2.25

mkdir build
cd build
../configure \
  --enable-shared --disable-static --disable-cli \
  --with-pic --with-gnu-ld \
  --prefix=$LIBNL_BASE

make -j3
make install

# rdma-core 25.0
# ==============

## exported environment
export RDMA_CORE_BASE=$CMSSW_BASE/local/rdma-core/25.0
export PATH=$RDMA_CORE_BASE/bin:$PATH
export LD_LIBRARY_PATH=$RDMA_CORE_BASE/lib64:$LD_LIBRARY_PATH

## download, configure, build and install
cd $CMSSW_BASE/build
git clone https://github.com/linux-rdma/rdma-core.git -b v25.0 rdma-core-25.0
cd rdma-core-25.0


mkdir build
cd build
PKG_CONFIG_PATH=$LIBNL_BASE/lib/pkgconfig \
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$RDMA_CORE_BASE \
  -DCMAKE_EXE_LINKER_FLAGS=-L$LIBNL_BASE/lib \
  -DCMAKE_SHARED_LINKER_FLAGS=-L$LIBNL_BASE/lib \
  -DPANDOC_EXECUTABLE= \
  -DRST2MAN_EXECUTABLE= \
  -DNO_PYVERBS=1 \
  -DENABLE_RESOLVE_NEIGH=0\
  ..


TMP=`mktemp -p .` && cat cmake_install.cmake | grep -v man/cmake_install.cmake > $TMP && mv $TMP cmake_install.cmake

make -j3
make install


# knem 1.1.3
# ==========


## exported environment
export KNEM_BASE=$CMSSW_BASE/local/knem/1.1.3

## download, configure, build and install
cd $CMSSW_BASE/build
wget http://gforge.inria.fr/frs/download.php/37186/knem-1.1.3.tar.gz
tar xaf knem-1.1.3.tar.gz
cd knem-1.1.3

mkdir build
cd build
../configure --prefix=$KNEM_BASE
make -j3
make install

# xpmem
# =====

## exported environment
export XPMEM_DATE=20180414
export XPMEM_HASH=bae6eea4eb6f7cf88e5de0d197b78efab7136f8c
export XPMEM_BASE=$CMSSW_BASE/local/xpmem/$XPMEM_DATE
export LD_LIBRARY_PATH=$XPMEM_BASE/lib:$LD_LIBRARY_PATH


## download, configure, build and install
cd $CMSSW_BASE/build
git clone https://gitlab.com/hjelmn/xpmem.git xpmem-$XPMEM_DATE
cd xpmem-$XPMEM_DATE
git reset --hard $XPMEM_HASH

./autogen.sh
./configure \
  --with-pic --with-gnu-ld \
  --prefix=$XPMEM_BASE

make -j3
make install

# gdrcopy 1.3     #I could not solve the errors here.
# ===========

## exported environment
export GDRCOPY_BASE=$CMSSW_BASE/local/gdrcopy/1.3
export LD_LIBRARY_PATH=$GDRCOPY_BASE/lib64:$LD_LIBRARY_PATH

## download, configure, build and install
cd $CMSSW_BASE/build
git clone https://github.com/NVIDIA/gdrcopy.git -b v1.3 gdrcopy-1.3
cd gdrcopy-1.3

sed -i -e's/^LDFLAGS.*/& -L$(CUDA)\/drivers/' Makefile
mkdir -p $GDRCOPY_BASE/lib64 $GDRCOPY_BASE/include
make CUDA=$CUDA_BASE config lib exes 
make PREFIX=$GDRCOPY_BASE install


# UCX 1.6.0
# =========

## exported environment
export UCX_BASE=$CMSSW_BASE/local/ucx/1.6.0
export PATH=$UCX_BASE/bin:$PATH
export LD_LIBRARY_PATH=$UCX_BASE/lib:$LD_LIBRARY_PATH

## download, configure, build and install
cd $CMSSW_BASE/build
git clone https://github.com/openucx/ucx.git -b v1.6.0 ucx-1.6.0

cd ucx-1.6.0
./autogen.sh
mkdir build
cd build

CPPFLAGS="-I$NUMACTL_BASE/include -I$RDMA_CORE_BASE/include" \
LDFLAGS="-L$CUDA_BASE/drivers -L$NUMACTL_BASE/lib -L$RDMA_CORE_BASE/lib64" \
LIBS="-libverbs" \
../configure \
  --enable-optimizations \
  --disable-logging \
  --enable-debug \
  --disable-assertions \
  --disable-params-check \
  --enable-cma \
  --enable-mt \
  --enable-devel-headers \
  --prefix=$UCX_BASE \
  --with-pic \
  --with-gnu-ld \
  --with-avx \
  --with-sse41 \
  --with-sse42 \
  --with-cuda=$CUDA_BASE \
  --without-rocm \
  --with-gdrcopy=$GDRCOPY_BASE \
  --with-verbs=$RDMA_CORE_BASE \
  --with-mlx5-dv \
  --with-rc \
  --with-ud \
  --with-dc \
  --with-ib-hw-tm \
  --with-dm \
  --without-cm \
  --with-rdmacm=$RDMA_CORE_BASE \
  --with-knem=$KNEM_BASE \
  --with-xpmem=$XPMEM_BASE \
  --without-ugni

make -j3
make install

# libevent
# ========

# exported environment
export LIBEVENT_BASE=$CMSSW_BASE/local/libevent/2.1.11
export PATH=$LIBEVENT_BASE/bin:$PATH
export LD_LIBRARY_PATH=$LIBEVENT_BASE/lib:$LD_LIBRARY_PATH

#download, configure, build and install
cd $CMSSW_BASE/build
wget https://github.com/libevent/libevent/releases/download/release-2.1.11-stable/libevent-2.1.11-stable.tar.gz
tar xaf libevent-2.1.11-stable.tar.gz
cd libevent-2.1.11-stable

mkdir build
cd build
../configure --prefix=$LIBEVENT_BASE --disable-samples --enable-shared --with-pic --with-gnu-ld
make -j3
make install


# Open MPI 4.0.x
# ==============

# exported environment
export OPENMPI_DATE=20190802
export OPENMPI_HASH=e547a2b94d315e48bb32de950d58966289327348
export OPENMPI_BASE=$CMSSW_BASE/local/ompi/4.0.2a-$OPENMPI_DATE
export OPAL_PREFIX=$OPENMPI_BASE
export PATH=$OPENMPI_BASE/bin:$PATH
export LD_LIBRARY_PATH=$OPENMPI_BASE/lib:$LD_LIBRARY_PATH

# download, configure, build and install
cd $CMSSW_BASE/build
git clone https://github.com/open-mpi/ompi.git -b v4.0.x ompi-4.0.2a-$OPENMPI_DATE
cd ompi-4.0.2a-$OPENMPI_DATE
git reset --hard $OPENMPI_HASH
./autogen.pl

#wget https://download.open-mpi.org/release/open-mpi/v4.0/#openmpi-4.0.1.tar.gz
#tar -xaf openmpi-4.0.1.tar.gz
#cd openmpi-4.0.1/

mkdir build
cd build

LDFLAGS="-L$LIBNL_BASE/lib" \
../configure \
  --prefix=$OPENMPI_BASE \
  --with-pic \
  --with-gnu-ld \
  --enable-heterogeneous \
  --enable-ipv6 \
  --disable-mpi-interface-warning \
  --disable-mpi-fortran \
  --disable-mpi-cxx \
  --enable-cxx-exceptions \
  --enable-shared \
  --disable-static \
  --enable-oshmem \
  --enable-openib-rdmacm-ibaddr \
  --enable-install-libpmix \
  --with-libevent=$LIBEVENT_BASE \
  --with-libnl=$LIBNL_BASE \
  --with-zlib=$ZLIB_BASE \
  --with-cuda=$CUDA_BASE \
  --with-ucx=$UCX_BASE \
  --with-pmix=internal \
  --with-hwloc=internal \
  --with-xpmem=$XPMEM_BASE \
  --with-knem=$KNEM_BASE \
  --with-cma \
  --without-ugni
  #--with-ofi=$OFI_BASE \           // unbounded variable
  #--with-verbs=$RDMA_CORE_BASE \   // weird (syntax?) errors

make -j3
make install

# other flags we may need, but require additional libraries
#  --enable-hwloc-pci
#  --with-fca
#  --with-hcoll
#  --with-mxm