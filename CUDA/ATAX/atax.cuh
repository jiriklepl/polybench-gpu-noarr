#ifndef _ATAX_CUH
#define _ATAX_CUH

#include "defines.cuh"

#ifdef MINI_DATASET
# define NX 4096
# define NY 4096
#elif defined(SMALL_DATASET)
# define NX 4096
# define NY 4096
#elif defined(MEDIUM_DATASET)
# define NX 4096
# define NY 4096
#elif defined(LARGE_DATASET)
# define NX 4096
# define NY 4096
#elif defined(EXTRALARGE_DATASET)
# define NX 4096
# define NY 4096
#endif

#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#endif // _ATAX_CUH
