#!/bin/bash -e

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib64

# Create the build directory
cmake -E make_directory build
cd build

# Configure the build
cmake -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CUDA_FLAGS="-DMINI_DATASET -DDATA_TYPE_IS_FLOAT -O3" \
	-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++-14 \
	..

# Build the project
NPROC=$(nproc)
cmake --build . --config Release -j"$NPROC"
