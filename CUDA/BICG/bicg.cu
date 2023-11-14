#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/interop/serialize_data.hpp>

#include "defines.cuh"
#include "bicg.cuh"

using num_t = DATA_TYPE;

namespace {

void init(/* args */) {

}

void compute_bicg(/* args */) {

}

void compute_bicgCuda(/* args */) {

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
	compute_bicgCuda(/* args */);
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<long double>(end - start);

#ifdef DEBUG
	compute_bicg(/* args */);
#endif

	// print results
	if (argv[0] != ""s) {
		std::cout << std::fixed << std::setprecision(2);
		noarr::serialize_data(std::cout, G.get_ref() ^ noarr::hoist<'i'>());
	}

	// print time
	std::cerr << duration.count() << std::endl;
}
