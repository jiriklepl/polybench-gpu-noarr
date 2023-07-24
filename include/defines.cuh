#ifndef DEFINES_HPP
#define DEFINES_HPP

#define CUCH(status)  do { cudaError_t err = status; if (err != cudaSuccess) std::cerr << __FILE__ ":" << __LINE__ << ": error: " << cudaGetErrorString(err) << "\n\t" #status << std::endl, exit(err); } while (false)

#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
# error "Please define one of MINI_DATASET, SMALL_DATASET, MEDIUM_DATASET, LARGE_DATASET, EXTRALARGE_DATASET"
# define MINI_DATASET
#endif

#if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) && !defined(DATA_TYPE_IS_DOUBLE)
# error "Please define one of DATA_TYPE_IS_INT, DATA_TYPE_IS_FLOAT, DATA_TYPE_IS_DOUBLE"
# define DATA_TYPE_IS_FLOAT
#endif

#if defined(DATA_TYPE_IS_FLOAT)
# define DATA_TYPE float
#elif defined(DATA_TYPE_IS_DOUBLE)
# define DATA_TYPE double
#endif

#include <noarr/structures_extended.hpp>
#include <noarr/structures/interop/bag.hpp>

template<class Struct>
class managed_bag {
public:
    using value_type = noarr::scalar_t<Struct>;

    managed_bag(Struct layout) : _layout(layout), _data(nullptr) {
        CUCH(cudaMallocManaged(&_data, _layout | noarr::get_size()));
    }

    ~managed_bag() {
        CUCH(cudaFree(_data));
    }

    managed_bag(const managed_bag&) = delete;
    managed_bag& operator=(const managed_bag&) = delete;

    auto get_ref() noexcept {
        return noarr::make_bag(_layout, _data);
    }

private:
    Struct _layout;
    value_type* _data;
};

#endif // DEFINES_HPP
