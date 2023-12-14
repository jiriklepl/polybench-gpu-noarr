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

constexpr num_t ALPHA = 32412.0f;
constexpr num_t BETA = 2123.0f;

// initialize data
void init(auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j

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

template<class inner_t, class C_t, class A_t, class B_t>
__global__ void kernel_gemm(inner_t inner, C_t C, A_t A, B_t B) {
	// C: i x j
	// A: i x k
	// B: k x j

	inner.template for_dims<'s', 't'>([=](auto inner) {
		auto state = inner.state();
		C[state] *= BETA;

		inner.template for_each<'k'>([=](auto state) {
			C[state] += ALPHA * A[state] * B[state];
		});
	});
}

// run kernels
void run_gemm(auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j
	auto trav = noarr::traverser(C, A, B)
		.order(noarr::into_blocks_dynamic<'i', 'I', 'i', 's'>(DIM_THREAD_BLOCK_X))
		.order(noarr::into_blocks_dynamic<'j', 'J', 'j', 't'>(DIM_THREAD_BLOCK_Y));

	noarr::cuda_threads<'I', 'i', 'J', 'j'>(trav)
		.simple_run(kernel_gemm, 0, C, A, B);

	CUCH(cudaGetLastError()); // check for configuration errors
	CUCH(cudaDeviceSynchronize()); // join, check for execution errors
}

class experiment : public virtual_experiment {
	template<class A, class B, class C>
	struct experiment_data : virtual_data {
		C c;
		A a;
		B b;

		experiment_data(C c, A a, B b)
			: c(std::move(c)), a(std::move(a)), b(std::move(b)) { }

		void run() override {
			run_gemm(c.get_device_ref(), a.get_device_ref(), b.get_device_ref());
		}

		void print_results(std::ostream& os) override {
			c.fetch_to_host();
			noarr::serialize_data(os, c.get_host_ref() ^ noarr::hoist<'i'>());
		}
	};

public:
	experiment() {
		std::size_t ni = NI;
		std::size_t nj = NJ;
		std::size_t nk = NK;

		cudaInit();

		experiment_data new_data{
			managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj)),
			managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nk)),
			managed_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'k', 'j'>(nk, nj)),
		};

		// initialize data
		init(new_data.c.get_host_ref(), new_data.a.get_host_ref(), new_data.b.get_host_ref());

		new_data.a.fetch_to_device();
		new_data.b.fetch_to_device();
		new_data.c.fetch_to_device();

		data = std::make_unique<decltype(new_data)>(std::move(new_data));
	}
};


} // namespace

REGISTER_EXPERIMENT(gemm);
