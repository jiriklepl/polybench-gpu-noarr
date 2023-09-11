#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "common.hpp"

int main(int argc, char** argv) {
	using namespace std::string_literals;

	auto experiment = make_experiment(argc, argv);

	auto start = std::chrono::high_resolution_clock::now();

	// run kernels
	experiment->run();

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration<double>(end - start);

	// print results
	if (argv[0] != ""s) {
        std::cout << std::fixed << std::setprecision(2);
		experiment->print_results(std::cout);
	}

	std::cerr << duration.count() << std::endl;
}
