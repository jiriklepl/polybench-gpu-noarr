#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.cuh"
#include "3mm.cuh"

using num_t = DATA_TYPE;

namespace {

noarr::dim<0> i_guard;
noarr::dim<1> j_guard;
noarr::dim<2> k_guard;
noarr::dim<3> m_guard;
noarr::dim<4> l_guard;

void init(auto A, auto B, auto C, auto D) {
	// A: i x k
	// B: k x j
	// C: j x m
	// D: m x l

	auto ni = A | noarr::get_length<'i'>();
	auto nj = B | noarr::get_length<'j'>();
	auto nk = A | noarr::get_length<'k'>();
	auto nl = D | noarr::get_length<'l'>();

	noarr::traverser(A)
		.for_each([=](auto state) {
			auto [i, k] = noarr::get_indices<'i', 'k'>(state);
			A[state] = ((num_t)i * k) / ni;
		});

	noarr::traverser(B)
		.for_each([=](auto state) {
			auto [k, j] = noarr::get_indices<'k', 'j'>(state);
			B[state] = ((num_t)k * (j + 1)) / nj;
		});

	noarr::traverser(C)
		.for_each([=](auto state) {
			auto [j, m] = noarr::get_indices<'j', 'm'>(state);
			C[state] = ((num_t)j * (m + 3)) / nl;
		});

	noarr::traverser(D)
		.for_each([=](auto state) {
			auto [m, l] = noarr::get_indices<'m', 'l'>(state);
			D[state] = ((num_t)m * (l + 2)) / nk;
		});
}

// __global__ void kernel_3mm_1(auto traverser, auto A, auto B, auto E) {
// 	// A: i x k
// 	// B: k x j
// 	// E: i x j

// 	traverser.template for_dims<i_guard, j_guard>([=](auto inner) {
// 		auto state = inner.state();
// 		E[state] = 0;

// 		inner.template for_each<'k'>([=](auto state) {
// 			E[state] += A[state] * B[state];
// 		});
// 	});
// }

// __global__ void kernel_3mm_2(auto traverser, auto C, auto D, auto F) {
// 	// C: j x m
// 	// D: m x l
// 	// F: j x l

// 	traverser.template for_dims<j_guard, l_guard>([=](auto inner) {
// 		auto state = inner.state();
// 		F[state] = 0;

// 		inner.template for_each<'m'>([=](auto state) {
// 			F[state] += C[state] * D[state];
// 		});
// 	});
// }

// __global__ void kernel_3mm_3(auto traverser, auto E, auto F, auto G) {
// 	// E: i x j
// 	// F: j x l
// 	// G: i x l

// 	traverser.template for_dims<i_guard, l_guard>([=](auto inner) {
// 		auto state = inner.state();
// 		G[state] = 0;

// 		inner.template for_each<'m'>([=](auto state) {
// 			G[state] += E[state] * F[state];
// 		});
// 	});
// }

void compute_3mm(/* args */) {

}

void compute_3mmCuda(auto A, auto B, auto C, auto D, auto E, auto F, auto G) {
	{
		noarr::traverser(A, B, E)
			.order(noarr::into_blocks_dynamic<'i', 'I', 'i', i_guard>(DIM_THREAD_BLOCK_X))
			.order(noarr::into_blocks_dynamic<'j', 'J', 'j', j_guard>(DIM_THREAD_BLOCK_Y))
			.template for_dims<'i', 'j', 'I', 'J'>([=](auto inner) {
				inner.template for_each<i_guard, j_guard>([=](auto state) {
				});
		});
	}
}

} // namespace

int main(int argc, char** argv) {
	using namespace std::string_literals;

	// problem size
	std::size_t ni = NI;
	std::size_t nj = NJ;
	std::size_t nk = NK;
	std::size_t nl = NL;
	std::size_t nm = NM;


	// data
	auto E = make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'j'>(ni, nj));
	auto A = make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'k'>(ni, nk));
	auto B = make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'k', 'j'>(nk, nj));

	auto F = make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'j', 'l'>(nj, nl));
	auto C = make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'j', 'm'>(nj, nm));
	auto D = make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'m', 'l'>(nm, nl));

	auto G = make_bag(noarr::scalar<num_t>() ^ noarr::sized_vectors<'i', 'l'>(ni, nl));

	// initialize data
	init(A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref());

	// run computation
	auto start = std::chrono::high_resolution_clock::now();
	compute_3mmCuda(A.get_ref(), B.get_ref(), C.get_ref(), D.get_ref(), E.get_ref(), F.get_ref(), G.get_ref());
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

	// print results
	if (argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, G.get_ref() ^ noarr::hoist<'i'>());
	}

	// print time
	std::cerr << duration.count() << std::endl;
}
