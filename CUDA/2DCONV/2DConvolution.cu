#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>
#include <noarr/structures/interop/cuda_traverser.cuh>

#include "defines.cuh"
#include "2DConvolution.cuh"

using num_t = DATA_TYPE;

namespace {

constexpr noarr::dim<__LINE__> i_guard;
constexpr noarr::dim<__LINE__> j_guard;


// initialize data
void init(auto A) {
	// A: i x j

	noarr::traverser(A).for_each([=](auto state) {
		A[state] = (float)rand() / RAND_MAX;
	});
}

__global__ void kernel_2dconv(auto inner, auto A, auto B) {
	// A: i x j
	// B: i x j
	using noarr::neighbor;

	num_t c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	inner.template for_each<i_guard, j_guard>([=](auto state) {
		B[state] = c11 * A[neighbor<'i', 'j'>(state, -1, 1)] +
			c21 * A[neighbor<'i', 'j'>(state, -1, 0)] +
			c31 * A[neighbor<'i', 'j'>(state, -1, +1)] +
			c12 * A[neighbor<'i', 'j'>(state, 0, -1)] +
			c22 * A[neighbor<'i', 'j'>(state, 0, 0)] +
			c32 * A[neighbor<'i', 'j'>(state, 0, +1)] +
			c13 * A[neighbor<'i', 'j'>(state, +1, -1)] +
			c23 * A[neighbor<'i', 'j'>(state, +1, 0)] +
			c33 * A[neighbor<'i', 'j'>(state, +1, +1)];
	});
}

// run kernels
void run_2dconv(auto A, auto B) {
	// A: i x j
	// B: i x j
	auto trav = noarr::traverser(A, B)
		.order(noarr::symmetric_spans<'i', 'j'>(A, 1, 1))
		.order(noarr::into_blocks_dynamic<'i', 'I', 'i', i_guard>(DIM_THREAD_BLOCK_X))
		.order(noarr::into_blocks_dynamic<'j', 'J', 'j', j_guard>(DIM_THREAD_BLOCK_Y));

	noarr::cuda_threads<'I', 'i', 'J', 'j'>(trav)
		.simple_run(kernel_2dconv, 0, A, B);

	CUCH(cudaGetLastError()); // check for configuration errors
	CUCH(cudaDeviceSynchronize()); // join, check for execution errors
}

} // namespace

int main(int argc, char** argv) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;

	// data
	auto A = managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));
	auto B = managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));

	init(A.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernels
	run_2dconv(A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
