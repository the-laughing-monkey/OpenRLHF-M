#!/bin/bash

# Simple script to verify MPI and nccl-tests setup

# Test if MPI compiler (mpicc) is in PATH
if ! command -v mpicc &> /dev/null; then
    echo "Error: mpicc not found. Please install an MPI implementation (e.g., OpenMPI or MPICH) and ensure mpicc is in your PATH."
    exit 1
fi

# Create a temporary MPI test program
echo "Compiling a simple MPI test program..."
cat << 'EOF' > test_mpi.c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
        printf("MPI test program running successfully.\n");
    }
    MPI_Finalize();
    return 0;
}
EOF

# Compile the test program
mpicc test_mpi.c -o test_mpi
if [ $? -ne 0 ]; then
    echo "Failed to compile the MPI test program. Please check your MPI installation and development headers (mpi.h)."
    rm -f test_mpi.c test_mpi
    exit 1
fi

# Run the test program
./test_mpi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "The MPI test program failed to run. There might be issues with your MPI environment."
    rm -f test_mpi.c test_mpi
    exit 1
else
    echo "MPI test program compiled and ran successfully."
fi

rm -f test_mpi.c test_mpi

# Check and clone the nccl-tests repository if not present
if [ ! -d "nccl-tests" ]; then
    echo "Cloning nccl-tests repository..."
    git clone https://github.com/NVIDIA/nccl-tests.git
    if [ $? -ne 0 ]; then
         echo "Failed to clone nccl-tests repository."
         exit 1
    fi
else
    echo "nccl-tests repository already exists."
fi

# Build nccl-tests
cd nccl-tests || { echo "Failed to enter nccl-tests directory."; exit 1; }
MPI_INCLUDES="-I/usr/lib/x86_64-linux-gnu/openmpi/include"
echo "Using MPI include flags: $MPI_INCLUDES"

echo "Building nccl-tests..."
make MPI=1 NCCL_HOME=/usr/local/nccl MPI_INCLUDES="$MPI_INCLUDES" NVCCFLAGS="-ccbin mpicc"
if [ $? -ne 0 ]; then
    echo "Failed to build nccl-tests. Check the build logs for errors."
    exit 1
fi

# Verify the build output
if [ ! -f "build/all_reduce_perf" ]; then
    echo "nccl-tests build doesn't contain the expected binary: build/all_reduce_perf"
    exit 1
fi

echo "nccl-tests compiled successfully. Your environment is ready to run the NCCL Ray Test script." 