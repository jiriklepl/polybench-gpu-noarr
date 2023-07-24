cmake_minimum_required(VERSION 3.10)
Include(FetchContent)

# set the project name
project(
  NoarrPolybenchGPU VERSION 0.0.1
  LANGUAGES CXX CUDA
)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# download the Noarr Structures library
FetchContent_Declare(
  Noarr
  GIT_REPOSITORY https://github.com/jiriklepl/noarr-structures.git
  GIT_TAG        constraints-WIP)
FetchContent_MakeAvailable(Noarr)

add_executable(gemm CUDA/GEMM/gemm.cu)
target_include_directories(gemm PRIVATE include)
target_include_directories(gemm PRIVATE ${Noarr_SOURCE_DIR}/include)
target_compile_options(gemm PRIVATE --expt-relaxed-constexpr)

add_executable(jacobi1d CUDA/JACOBI1D/jacobi1d.cu)
target_include_directories(jacobi1d PRIVATE include)
target_include_directories(jacobi1d PRIVATE ${Noarr_SOURCE_DIR}/include)
target_compile_options(jacobi1d PRIVATE --expt-relaxed-constexpr)

add_executable(2DConvolution CUDA/2DCONV/2DConvolution.cu)
target_include_directories(2DConvolution PRIVATE include)
target_include_directories(2DConvolution PRIVATE ${Noarr_SOURCE_DIR}/include)
target_compile_options(2DConvolution PRIVATE --expt-relaxed-constexpr)
