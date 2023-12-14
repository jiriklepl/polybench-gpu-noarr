#ifndef _ADI_CUH
#define _ADI_CUH

#include "defines.cuh"

#ifdef MINI_DATASET
# define TSTEPS 1
# define N 1024
#elif defined(SMALL_DATASET)
# define TSTEPS 1
# define N 1024
#elif defined(MEDIUM_DATASET)
# define TSTEPS 1
# define N 1024
#elif defined(LARGE_DATASET)
# define TSTEPS 1
# define N 1024
#elif defined(EXTRALARGE_DATASET)
# define TSTEPS 1
# define N 1024
#endif

#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#endif // _ADI_CUH
