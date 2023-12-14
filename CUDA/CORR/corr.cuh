#ifndef _CORR_CUH
#define _CORR_CUH

#include "defines.cuh"

#ifdef MINI_DATASET
#define M 2048
#define N 2048
#elif defined(SMALL_DATASET)
#define M 2048
#define N 2048
#elif defined(MEDIUM_DATASET)
#define M 2048
#define N 2048
#elif defined(LARGE_DATASET)
#define M 2048
#define N 2048
#elif defined(EXTRALARGE_DATASET)
#define M 2048
#define N 2048
#endif

/* Thread block dimensions for kernel 1*/
#define DIM_THREAD_BLOCK_KERNEL_1_X 256
#define DIM_THREAD_BLOCK_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2*/
#define DIM_THREAD_BLOCK_KERNEL_2_X 256
#define DIM_THREAD_BLOCK_KERNEL_2_Y 1

/* Thread block dimensions for kernel 3*/
#define DIM_THREAD_BLOCK_KERNEL_3_X 32
#define DIM_THREAD_BLOCK_KERNEL_3_Y 8

/* Thread block dimensions for kernel 4*/
#define DIM_THREAD_BLOCK_KERNEL_4_X 256
#define DIM_THREAD_BLOCK_KERNEL_4_Y 1

#endif // _CORR_CUH
