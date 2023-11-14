#ifndef _GEMM_HPP
#define _GEMM_HPP

#include <noarr/structures_extended.hpp>
#include <noarr/structures/interop/bag.hpp>

#ifdef MINI_DATASET
// # define ...
#elif defined(SMALL_DATASET)
// # define ...
#elif defined(MEDIUM_DATASET)
// # define ...
#elif defined(LARGE_DATASET)
// # define ...
#elif defined(EXTRALARGE_DATASET)
// # define ...
#endif

struct Gemm {
    void init();
    void compute();
    void print();
};

#endif // _GEMM_HPP
