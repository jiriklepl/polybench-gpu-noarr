cmake_minimum_required(VERSION 3.10)
Include(FetchContent)

# set the project name
project(
  NoarrPolybench VERSION 0.0.1
  DESCRIPTION "Polybench kernels implemented using Noarr Structures"
  LANGUAGES CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -allow-unsupported-compiler ")
set(CMAKE_CUDA_)


enable_language(CUDA)

# download testing library Catch2
FetchContent_Declare(
  Noarr
  GIT_REPOSITORY https://github.com/jiriklepl/noarr-structures.git
  GIT_TAG        main)
FetchContent_MakeAvailable(Noarr)

include_directories(include)
include_directories(${Noarr_SOURCE_DIR}/include)


# datamining
add_executable(correlation CUDA/CORR/corr.cu)
add_executable(covariance CUDA/COVAR/covar.cu)



# linear algebra - blas
add_executable(gemm CUDA/GEMM/gemm.cu)
add_executable(gemver CUDA/GEMVER/gemver.cu)
add_executable(gesummv CUDA/GESUMMV/gesummv.cu)
# TODO: add_executable(symm CUDA/SYMM/symm.cu)
add_executable(syr2k CUDA/SYR2K/syr2k.cu)
add_executable(syrk CUDA/SYRK/syrk.cu)
# TODO: add_executable(trmm CUDA/TRMM/trmm.cu)

# linear algebra - kernels
add_executable(2mm CUDA/2MM/2mm.cu)
add_executable(3mm CUDA/3MM/3mm.cu)
add_executable(atax CUDA/ATAX/atax.cu)
add_executable(bicg CUDA/BICG/bicg.cu)
add_executable(doitgen CUDA/DOITGEN/doitgen.cu)
add_executable(mvt CUDA/MVT/mvt.cu)

# linear algebra - solvers
# TODO: add_executable(cholesky CUDA/CHOLESKY/cholesky.cu)
# TODO: add_executable(durbin CUDA/DURBIN/durbin.cu)
add_executable(gramschmidt CUDA/GRAMSCHM/gramschm.cu)
add_executable(lu CUDA/LU/lu.cu)
# TODO: add_executable(ludcmp CUDA/LUDCMP/ludcmp.cu)
# TODO: add_executable(trisolv CUDA/TRISOLV/trisolv.cu)

# medley
# TODO: add_executable(deriche CUDA/DERICHE/deriche.cu)
# TODO: add_executable(floyd-warshall CUDA/FLOYD-WARSHALL/floyd-warshall.cu)
# TODO: add_executable(nussinov CUDA/NUSSINOV/nussinov.cu)

# stencils
add_executable(adi CUDA/ADI/adi.cu)
add_executable(fdtd-2d CUDA/FDTD-2D/fdtd-2d.cu)
# TODO: add_executable(heat-3d CUDA/HEAT-3D/heat-3d.cu)
add_executable(jacobi-1d CUDA/JACOBI1D/jacobi1d.cu)
add_executable(jacobi-2d CUDA/JACOBI2D/jacobi2d.cu)
# TODO: add_executable(seidel-2d CUDA/SEIDEL-2D/seidel-2d.cu)
