#include "neurstore/compress/index/router/single_index_router.h"


template<typename T>
int SingleIndexRouter<T>::route(const T &vector) {
    std::lock_guard lock(mutex_);
    // check if the current index size exceeds the maximum index size
    if (current_element_count_ + 1 > max_elements_per_index_) {
        current_index_id_++;
        current_element_count_ = 0;
    }
    return current_index_id_;
}

template<typename T>
void SingleIndexRouter<T>::add(const T &vector) {
    std::lock_guard lock(mutex_);
    current_element_count_ += 1;
}

template<typename T>
void SingleIndexRouter<T>::update(int index_id, const T &vector, int num_of_vectors) {
    std::lock_guard lock(mutex_);
    current_element_count_ += 1;
}

template<typename T>
void SingleIndexRouter<T>::serialize(std::ostream &out) const {
    std::lock_guard lock(mutex_);
    const int dimension = this->getDimension();
    const int n = this->getNumOfIndex();
    const int bit_width = this->getBitWidth();
    const int max_elements_per_index = max_elements_per_index_;
    const int current_index_id = current_index_id_;
    const int current_element_count = current_element_count_;

    out.write(reinterpret_cast<const char *>(&dimension), sizeof(int));
    out.write(reinterpret_cast<const char *>(&n), sizeof(int));
    out.write(reinterpret_cast<const char *>(&bit_width), sizeof(int));
    out.write(reinterpret_cast<const char *>(&max_elements_per_index), sizeof(int));
    out.write(reinterpret_cast<const char *>(&current_index_id), sizeof(int));
    out.write(reinterpret_cast<const char *>(&current_element_count), sizeof(int));

    if (!out) {
        throw std::runtime_error("SingleIndexRouter::serialize: failed to write to stream");
    }
}

template<typename T>
std::shared_ptr<SingleIndexRouter<T> > SingleIndexRouter<T>::deserialize(const uint8_t *data, size_t size) {
    const uint8_t *ptr = data;
    int dimension, nindex, bit_width, max_elements_per_index, current_index_id, current_element_count;
    std::memcpy(&dimension, ptr, sizeof(int));
    ptr += sizeof(int);
    std::memcpy(&nindex, ptr, sizeof(int));
    ptr += sizeof(int);
    std::memcpy(&bit_width, ptr, sizeof(int));
    ptr += sizeof(int);
    std::memcpy(&max_elements_per_index, ptr, sizeof(int));
    ptr += sizeof(int);
    std::memcpy(&current_index_id, ptr, sizeof(int));
    ptr += sizeof(int);
    std::memcpy(&current_element_count, ptr, sizeof(int));

    auto router = std::make_shared<SingleIndexRouter<T>>(
        dimension, max_elements_per_index, nindex, current_index_id, current_element_count, bit_width
    );
    return router;
}

template class SingleIndexRouter<TensorType::VectorFloat64>;
