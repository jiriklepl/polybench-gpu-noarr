#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>

#include "defines.cuh"
#include "gemm.hpp"

using num_t = DATA_TYPE;

namespace {

void init(/* args */) {

}

void compute_gemm(/* args */) {

}

void compute_gemmCuda(/* args */) {

}

}

int main(int argc, char** argv) {
	using namespace std::string_literals;

	// problem size
	// ...

	// data
	// ...

	// initialize data
	init(/* args */);

	// run computation
	auto start = std::chrono::high_resolution_clock::now();
	compute_gemmCuda(/* args */);
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

#ifdef DEBUG
	compute_gemm(/* args */);
#endif

	// print results
	if (argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		// noarr::serialize_data(std::cout, G.get_ref() ^ noarr::hoist<'i'>());
	}

	// print time
	std::cerr << duration.count() << std::endl;
}
