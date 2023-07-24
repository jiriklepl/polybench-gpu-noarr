#!/bin/bash

# Create the build directory
cmake -E make_directory build || exit 1
cd build || exit 1

# Configure the build
cmake -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CUDA_FLAGS="-DMINI_DATASET -DDATA_TYPE_IS_FLOAT" \
	.. \
	|| exit 1

# Build the project
NPROC=$(nproc)
cmake --build . --config Release -j"$NPROC" || exit 1
