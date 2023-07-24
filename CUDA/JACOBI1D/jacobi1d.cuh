#ifndef _JACOBI1D_CUH
#define _JACOBI1D_CUH

#include "defines.cuh"

#ifdef MINI_DATASET
# define TSTEPS 10000
# define N 4096
#elif defined(SMALL_DATASET)
# define TSTEPS 10000
# define N 4096
#elif defined(MEDIUM_DATASET)
# define TSTEPS 10000
# define N 4096
#elif defined(LARGE_DATASET)
# define TSTEPS 10000
# define N 4096
#elif defined(EXTRALARGE_DATASET)
# define TSTEPS 10000
# define N 4096
#endif

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256

#endif // _JACOBI1D_CUH
