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

tmpdir=$(mktemp -d)

trap "echo deleting $tmpdir; rm -rf $tmpdir" EXIT

( cd "$POLYBENCH_GPU_DIR/CUDA" && bash compileCodes.sh ) || exit 1
( cd . && ./build.sh ) || exit 1

compare_algorithms() {
	echo "Running $1" >&2


	printf "\tnoarr: " >&2
    "$BUILD_DIR/runner" "$1" >/dev/null 2>&1
    "$BUILD_DIR/runner" "$1" 2>&1 >"$tmpdir/noarr.log" | grep -oE "[0-9]+\.[0-9]{2,}" >&2

	printf "\tbaseline: " >&2
	"$2" >/dev/null 2>&1
    "$2" 2>"$tmpdir/baseline.log" | grep -oE "[0-9]+\.[0-9]{2,}" >&2

	paste <(grep -oE '[0-9]+\.[0-9]+' "$tmpdir/baseline.log") <(grep -oE '[0-9]+(\.[0-9]+)?' "$tmpdir/noarr.log") |
	awk "BEGIN {
		different = 0
		n = 0
		changes = 0
	}

	{
		n++
		if (\$1 != \$2 && changes < 10) {
			print \"baseline\", n, \$1
			print \"   noarr\", n, \$2
			changes++
			different = 1
		}

		if (changes >= 10)
			nextfile

		next
	}

	{ different = 1; nextfile }

	END {
		if (different) {
			printf \"Different output on %s \n\", \"$1\"
			exit 1
		}
	}" 1>&2
}

compare_algorithms gemm "$POLYBENCH_GPU_DIR/CUDA/GEMM/gemm.exe"
compare_algorithms 2mm "$POLYBENCH_GPU_DIR/CUDA/2MM/2mm.exe"
compare_algorithms 2DConvolution "$POLYBENCH_GPU_DIR/CUDA/2DCONV/2DConvolution.exe"
compare_algorithms gramschmidt "$POLYBENCH_GPU_DIR/CUDA/GRAMSCHM/gramschmidt.exe"
compare_algorithms jacobi2d "$POLYBENCH_GPU_DIR/CUDA/JACOBI2D/jacobi2D.exe"
