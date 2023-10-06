cmake_minimum_required(VERSION 3.10)
Include(FetchContent)

# set the project name
project(
  NoarrPolybenchGPU VERSION 0.0.1
  LANGUAGES CXX
)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

enable_language(CUDA)

# download the Noarr Structures library
FetchContent_Declare(
  Noarr
  GIT_REPOSITORY https://github.com/jiriklepl/noarr-structures.git
  GIT_TAG        multimerge)
FetchContent_MakeAvailable(Noarr)

include_directories(include)
include_directories(${Noarr_SOURCE_DIR}/include)

add_library(gemm_kernel STATIC CUDA/GEMM/gemm.cu)
target_compile_options(gemm_kernel PRIVATE --expt-relaxed-constexpr)

add_executable(gemm CPP/main.cpp)
target_link_libraries(gemm gemm_kernel)

# add_executable(jacobi1d CUDA/JACOBI1D/jacobi1d.cu)
# target_include_directories(jacobi1d PRIVATE include)
# target_include_directories(jacobi1d PRIVATE ${Noarr_SOURCE_DIR}/include)
# target_compile_options(jacobi1d PRIVATE --expt-relaxed-constexpr)

# add_executable(2DConvolution CUDA/2DCONV/2DConvolution.cu)
# target_include_directories(2DConvolution PRIVATE include)
# target_include_directories(2DConvolution PRIVATE ${Noarr_SOURCE_DIR}/include)
# target_compile_options(2DConvolution PRIVATE --expt-relaxed-constexpr)
