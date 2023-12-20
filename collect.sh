#!/usr/bin/env bash


BUILD_DIR=${BUILD_DIR:-build}
SKIP_DIFF=${SKIP_DIFF:-0}

if [ -z "$POLYBENCH_GPU_DIR" ]; then
	POLYBENCH_GPU_DIR="$BUILD_DIR/PolyBenchGpu"
	mkdir -p "$POLYBENCH_GPU_DIR" || exit 1
	if [ -d "$POLYBENCH_GPU_DIR/.git" ]; then
		( cd "$POLYBENCH_GPU_DIR" && git fetch && git checkout noarr-compare && git pull ) || exit 1
	else
		git clone --branch=noarr-compare "https://github.com/jiriklepl/PolyBenchGpu.git" "$POLYBENCH_GPU_DIR" || exit 1
	fi
fi

dirname=$(mktemp -d)

trap "echo deleting $dirname; rm -rf $dirname" EXIT

( cd "$POLYBENCH_GPU_DIR/CUDA" && srun -A kdss -p gpu-short --exclusive -wampere01 --gres=gpu bash compileCodes.sh ) || exit 1
( cd . && srun -A kdss -p gpu-short --exclusive -wampere01 --gres=gpu ./build.sh ) || exit 1

DATA_DIR="data"

mkdir -p "$DATA_DIR"

compare_algorithms() {
    echo "collecting $1"
    ( srun -A kdss -p gpu-short --exclusive -wampere01 --gres=gpu ./run_noarr_algorithm.sh "Noarr" "$1" & wait ) > "$DATA_DIR/$1.log"
	echo "" >> "$DATA_DIR/$1.log"
    ( srun -A kdss -p gpu-short --exclusive -wampere01 --gres=gpu ./run_baseline_algorithm.sh "Baseline" "$2" & wait ) >> "$DATA_DIR/$1.log"
    echo "done"
}

compare_algorithms gemm "$POLYBENCH_GPU_DIR/CUDA/GEMM/gemm.exe"
compare_algorithms 2mm "$POLYBENCH_GPU_DIR/CUDA/2MM/2mm.exe"
compare_algorithms 2DConvolution "$POLYBENCH_GPU_DIR/CUDA/2DCONV/2DConvolution.exe"
compare_algorithms gramschmidt "$POLYBENCH_GPU_DIR/CUDA/GRAMSCHM/gramschmidt.exe"
compare_algorithms jacobi2d "$POLYBENCH_GPU_DIR/CUDA/JACOBI2D/jacobi2D.exe"
