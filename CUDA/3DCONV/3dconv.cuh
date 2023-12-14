#ifndef _3DCONV_CUH
#define _3DCONV_CUH

#include "defines.cuh"

#ifdef MINI_DATASET
# define NI 256
# define NJ 256
# define NK 256
#elif defined(SMALL_DATASET)
# define NI 256
# define NJ 256
# define NK 256
#elif defined(MEDIUM_DATASET)
# define NI 256
# define NJ 256
# define NK 256
#elif defined(LARGE_DATASET)
# define NI 256
# define NJ 256
# define NK 256
#elif defined(EXTRALARGE_DATASET)
# define NI 256
# define NJ 256
# define NK 256
#endif

#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

#endif // _3DCONV_CUH
