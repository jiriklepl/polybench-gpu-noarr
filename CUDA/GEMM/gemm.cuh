#ifndef _GEMM_CUH
#define _GEMM_CUH

#include "defines.cuh"

#ifdef MINI_DATASET
# define NI 512
# define NJ 512
# define NK 512
#elif defined(SMALL_DATASET)
# define NI 512
# define NJ 512
# define NK 512
#elif defined(MEDIUM_DATASET)
# define NI 512
# define NJ 512
# define NK 512
#elif defined(LARGE_DATASET)
# define NI 512
# define NJ 512
# define NK 512
#elif defined(EXTRALARGE_DATASET)
# define NI 512
# define NJ 512
# define NK 512
#endif

#endif // _GEMM_CUH
