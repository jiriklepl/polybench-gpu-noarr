#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>
#include <noarr/structures/interop/cuda_traverser.cuh>

#include "defines.cuh"
#include "gemm.cuh"

using num_t = DATA_TYPE;

namespace {

constexpr noarr::dim<__LINE__> i_guard;
constexpr noarr::dim<__LINE__> j_guard;

// initialize data
void init(num_t& alpha, num_t& beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j

	alpha = 32412;
	beta = 2123;

	noarr::traverser(A)
		.for_each([=](auto state) {
			auto [i, k] = noarr::get_indices<'i', 'k'>(state);

			A[state] = ((num_t) i * k) / NI;
		});
		
	noarr::traverser(B)
		.for_each([=](auto state) {
			auto [k, j] = noarr::get_indices<'k', 'j'>(state);

			B[state] = ((num_t) k * j) / NI;
		});
	
	noarr::traverser(C)
		.for_each([=](auto state) {
			auto [i, j] = noarr::get_indices<'i', 'j'>(state);

			C[state] = ((num_t) i * j) / NI;
		});
}

__global__ void kernel_gemm(auto inner, num_t alpha, num_t beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j

	inner.template for_dims<i_guard, j_guard>([=](auto inner) {
		auto state = inner.state();
		C[state] *= beta;

		inner.template for_each<'k'>([=](auto state) {
			C[state] += alpha * A[state] * B[state];
		});
	});
}

// run kernels
void run_gemm(num_t alpha, num_t beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j

	auto trav = noarr::traverser(C, A, B)
		.order(noarr::into_blocks_dynamic<'i', 'I', 'i', i_guard>(DIM_THREAD_BLOCK_X))
		.order(noarr::into_blocks_dynamic<'j', 'J', 'j', j_guard>(DIM_THREAD_BLOCK_Y));

	noarr::cuda_threads<'I', 'i', 'J', 'j'>(trav)
		.simple_run(kernel_gemm, 0, alpha, beta, C, A, B);
	CUCH(cudaGetLastError()); // check for configuration errors
	CUCH(cudaDeviceSynchronize()); // join, check for execution errors
}

} // namespace

int main(int argc, char** argv) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;
	std::size_t nk = NK;

	// data
	num_t alpha;
	num_t beta;

	auto C = managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));
	auto A = managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nk));
	auto B = managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'k', 'j'>(nk, nj));

	// initialize data
	init(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernels
	run_gemm(alpha, beta, C.get_ref(), A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<double>(end - start);

	// print results
	if (argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, C.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
