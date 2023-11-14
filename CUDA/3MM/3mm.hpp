#ifndef _3MM_CUH
#define _3MM_CUH

#ifdef MINI_DATASET
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512
#elif defined(SMALL_DATASET)
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512
#elif defined(MEDIUM_DATASET)
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512
#elif defined(LARGE_DATASET)
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512
#elif defined(EXTRALARGE_DATASET)
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512
#endif

#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

#endif // _3MM_CUH
