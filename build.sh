#!/bin/bash -e

# Create the build directory
cmake -E make_directory build
cd build

# Configure the build
cmake -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CUDA_FLAGS="-DMINI_DATASET -DDATA_TYPE_IS_FLOAT" \
	-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++ \
	..

# Build the project
NPROC=$(nproc)
cmake --build . --config Release -j"$NPROC"
