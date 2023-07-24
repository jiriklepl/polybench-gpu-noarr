#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>
#include <noarr/structures/interop/cuda_traverser.cuh>

#include "defines.cuh"
#include "jacobi1d.cuh"

using num_t = DATA_TYPE;

namespace {

constexpr noarr::dim<__LINE__> i_guard;

// initialize data
void init(auto A, auto B) {
	// A: i
	// B: i

	noarr::traverser(A, B).for_each([=](auto state){
		auto i = noarr::get_index<'i'>(state);
		A[state] = ((num_t) 4 * i + 10) / N;
		B[state] = ((num_t) 7 * i + 11) / N;
	});
}

__global__ void kernel1_jacobi1d(auto inner, auto A, auto B) {
	// A: i
	// B: i

	using noarr::neighbor;

	inner.template for_each<i_guard>([=](auto state) {
		B[state] = 0.33333 * (A[neighbor<'i'>(state, -1)] + A[state] + A[neighbor<'i'>(state, +1)]);
	});
}

__global__ void kernel2_jacobi1d(auto inner, auto A, auto B) {
	// A: i
	// B: i

	inner.template for_each<i_guard>([=](auto state) {
		A[state] = B[state];
	});
}

// run kernels
void run_jacobi1d(std::size_t tsteps, auto A, auto B) {
	// A: i
	// B: i
	
	auto trav = noarr::traverser(A, B);

	trav
		.order(noarr::bcast<'t'>(tsteps))
		.order(noarr::symmetric_span<'i'>(trav.top_struct(), 1))
		.order(noarr::into_blocks_dynamic<'i', 'I', 'i', i_guard>(DIM_THREAD_BLOCK_X))
		.template for_dims<'t'>([=](auto inner) {
		auto cutrav = noarr::cuda_threads<'I', 'i'>(inner);

		cutrav.simple_run(kernel1_jacobi1d, 0, A, B);
		CUCH(cudaGetLastError()); // check for configuration errors
		cutrav.simple_run(kernel2_jacobi1d, 0, A, B);
		CUCH(cudaGetLastError()); // check for configuration errors
	});

	CUCH(cudaDeviceSynchronize()); // join, check for execution errors
}

} // namespace

int main(int argc, char** argv) {
	using namespace std::string_literals;

	// problem size
	std::size_t tsteps = TSTEPS;
	std::size_t n = N;

	// data
	auto A = managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));
	auto B = managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vector<'i'>(n));

	// initialize data
	init(A.get_ref(), B.get_ref());

	auto start = std::chrono::high_resolution_clock::now();

	// run kernels
	run_jacobi1d(tsteps, A.get_ref(), B.get_ref());

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, A.get_ref() ^ noarr::hoist<'i'>());
	}

	std::cerr << duration.count() << std::endl;
}
