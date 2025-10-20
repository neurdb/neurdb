#include "neurstore/compress/index/router/kmeans_router.h"

#include "hnswlib/hnswlib.h"
#include "neurstore/compress/method/linear_quantization.h"


template<typename T>
int KMeansRouter<T>::route(const T &vector) {
    const int dim = this->getDimension();

    if (vector.size() != dim) {
        throw std::invalid_argument("KMeansRouter::route: vector dimension mismatch");
    }

    if (centroids_.empty()) {
        return 0;   // if there are no centroids, return the first index
    }

    float best_distance = std::numeric_limits<float>::max();
    int best_id = 0;
    const auto dim_sz = static_cast<size_t>(dim);

    for (int i = 0; i < static_cast<int>(centroids_.size()); ++i) {
        const auto &centroid_buffer = centroids_[i];

        float distance = hnswlib::L2Sqr(
            vector.data(),
            centroid_buffer.data(),
            &dim_sz
        );
        if (distance < best_distance) {
            best_distance = distance;
            best_id = i;
        }
    }
    return best_id;
}

template<typename T>
void KMeansRouter<T>::add(const T &vector) {
    const int dim = this->getDimension();

    if (vector.size() != dim) {
        throw std::invalid_argument("KMeansRouter::add: dimension mismatch");
    }
    std::vector<uint8_t> encoded_data(dim * this->getBitWidth() / 8);
    std::memcpy(encoded_data.data(), vector.data(), dim * this->getBitWidth() / 8);
    centroids_.emplace_back(std::move(encoded_data));

    this->incrementNumOfIndex();
}

template<typename T>
void KMeansRouter<T>::update(int index_id, const T &vector, int num_of_vectors) {
    if (index_id < 0 || index_id > this->getNumOfIndex()) {
        throw std::out_of_range("KMeansRouter::update: index_id out of range");
    }
    if (centroids_.empty()) {
        // there is no need to update centroids if there are no centroids
        // this usually happens when the there is only one index and there is no need to route
        return;
    }
    Eigen::Map<const T> old_vector(reinterpret_cast<const typename T::Scalar*>(centroids_[index_id].data()), this->getDimension());
    TensorType::VectorFloat32 updated_vector = (old_vector * num_of_vectors + vector) / (num_of_vectors + 1);
    std::memcpy(centroids_[index_id].data(), updated_vector.data(), this->getDimension() * sizeof(typename T::Scalar));
}

template<typename T>
void KMeansRouter<T>::serialize(std::ostream &out) const {
    const int dimension = this->getDimension();
    const int n = this->getNumOfIndex();
    const int bit_width = this->getBitWidth();

    out.write(reinterpret_cast<const char *>(&dimension), sizeof(int));
    out.write(reinterpret_cast<const char *>(&n), sizeof(int));
    out.write(reinterpret_cast<const char *>(&bit_width), sizeof(int));

    for (const auto &centroid_buf: centroids_) {
        if (centroid_buf.size() != dimension * bit_width / 8) {
            throw std::runtime_error("KMeansRouter::serialize: invalid centroid buffer size");
        }
        out.write(
            reinterpret_cast<const char *>(centroid_buf.data()),
            static_cast<std::streamsize>(centroid_buf.size())
        );
    }

    if (!out) {
        throw std::runtime_error("KMeansRouter::serialize: failed to write to stream");
    }
}

template<typename T>
std::shared_ptr<KMeansRouter<T> > KMeansRouter<T>::deserialize(const uint8_t *data, size_t size) {
    constexpr size_t header_bytes = sizeof(int) * 3;
    if (size < header_bytes) {
        throw std::runtime_error("KMeansRouter::deserialize: buffer too small for header");
    }

    const uint8_t *ptr = data;

    int dimension, nindex, bit_width;
    std::memcpy(&dimension, ptr, sizeof(int));
    ptr += sizeof(int);
    std::memcpy(&nindex, ptr, sizeof(int));
    ptr += sizeof(int);
    std::memcpy(&bit_width, ptr, sizeof(int));
    ptr += sizeof(int);

    if (dimension <= 0 || nindex < 0 ||
        (bit_width != 4 && bit_width != 8 && bit_width != 16 && bit_width != 32 && bit_width != 64)) {
        throw std::runtime_error("KMeansRouter::deserialize: invalid header values");
    }

    const size_t element_size = bit_width / 8;
    const size_t centroid_buf_size = dimension * element_size;
    const size_t expected_total_size = header_bytes + nindex * centroid_buf_size;

    if (size != expected_total_size) {
        throw std::runtime_error("KMeansRouter::deserialize: buffer size mismatch");
    }

    auto router = std::make_shared<KMeansRouter>(dimension, nindex, bit_width);

    for (int i = 0; i < nindex; ++i) {
        const uint8_t *start = ptr + i * centroid_buf_size;
        router->centroids_.emplace_back(start, start + centroid_buf_size);
    }

    return router;
}

template class KMeansRouter<TensorType::VectorFloat32>;
