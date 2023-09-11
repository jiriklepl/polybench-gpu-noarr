#include <iostream>

#include <memory>
#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>
#include <noarr/structures/interop/cuda_traverser.cuh>

#include "common.hpp"
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


class gemm_experiment : public virtual_experiment {
	template<class A, class B, class C>
	struct gemm_data : experiment_data {
		gemm_data(num_t alpha, num_t beta, C c, A a, B b)
			: alpha(alpha), beta(beta), c(std::move(c)), a(std::move(a)), b(std::move(b)) { }

		num_t alpha;
		num_t beta;

		C c;
		A a;
		B b;

		void run() override {
			run_gemm(alpha, beta, c.get_ref(), a.get_ref(), b.get_ref());
		}

		void print_results(std::ostream& os) override {
			noarr::serialize_data(os, c.get_ref() ^ noarr::hoist<'i'>());
		}
	};

public:
	gemm_experiment() {
		std::size_t ni = NI;
		std::size_t nj = NJ;
		std::size_t nk = NK;

		gemm_data new_data{
			(num_t)0,
			(num_t)0,
			managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj)),
			managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nk)),
			managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'k', 'j'>(nk, nj)),
		};

		// initialize data
		init(new_data.alpha, new_data.beta, new_data.c.get_ref(), new_data.a.get_ref(), new_data.b.get_ref());

		data = std::make_unique<decltype(new_data)>(std::move(new_data));
	}
};

std::unique_ptr<virtual_experiment> make_experiment(int, char *[]) {
	return std::make_unique<gemm_experiment>();
}
