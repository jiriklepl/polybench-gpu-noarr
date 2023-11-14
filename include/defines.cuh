#ifndef DEFINES_HPP
#define DEFINES_HPP

#include <noarr/structures_extended.hpp>
#include <noarr/structures/interop/bag.hpp>
#include <type_traits>
#include <cuda.h>

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, std::exit(err); } while (false)

template<class Struct>
class managed_bag {
public:
	using layout_type = std::remove_cv_t<Struct>;
	using value_type = noarr::scalar_t<Struct>;

	managed_bag(Struct s) : layout_(s), ptr_(nullptr) {
		CUCH(cudaMallocManaged(&ptr_, layout_ | noarr::get_size()));
	}

	~managed_bag() {
		struct finnally {
			~finnally() {
				ptr = nullptr;
			}

			value_type *ptr;
		} finnally{ptr_};

		CUCH(cudaFree(ptr_));
	}

	constexpr auto get_ref() const noexcept {
		return noarr::make_bag(layout_, ptr_);
	}

private:
	layout_type layout_;
	value_type *ptr_;
};

constexpr auto div_ceil(auto a, auto b) noexcept {
	return (a + b - 1) / b;
}

#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
# error "Please define one of MINI_DATASET, SMALL_DATASET, MEDIUM_DATASET, LARGE_DATASET, EXTRALARGE_DATASET"
# define MINI_DATASET
#endif

#if !defined(DATA_TYPE_IS_FLOAT) && !defined(DATA_TYPE_IS_DOUBLE)
# error "Please define one of DATA_TYPE_IS_FLOAT, DATA_TYPE_IS_DOUBLE"
# define DATA_TYPE_IS_FLOAT
#endif

#if defined(DATA_TYPE_IS_FLOAT)
# define DATA_TYPE float
#elif defined(DATA_TYPE_IS_DOUBLE)
# define DATA_TYPE double
#endif

#endif // DEFINES_HPP