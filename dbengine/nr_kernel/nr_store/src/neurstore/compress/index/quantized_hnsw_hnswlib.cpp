#include "neurstore/compress/index/quantized_hnsw_hnswlib.h"

#include <thread>
#include <sys/mman.h>

#include "neurstore/utils/global.h"
#include "neurstore/compress/method/linear_quantization.h"


QuantizedHNSWIndexHNSWLIB::QuantizedHNSWIndexHNSWLIB(
    int dimension,
    int64_t next_id,
    size_t max_elements
): SimilarityIndex(dimension),
   max_elements_(max_elements) {
    next_id_.store(next_id, std::memory_order_relaxed);
    auto [m, ef_construction, ef_search] = getHNSWParamsForDim(dimension);
    m_ = m;
    ef_construction_ = ef_construction;
    ef_search_ = ef_search;
    space_ = std::make_unique<hnswlib::QuantizedL2Space>(dimension);
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float> >(
        space_.get(),
        max_elements_,
        m_,
        ef_construction_
    );
    index_->setEf(ef_search_);
}

QuantizedHNSWIndexHNSWLIB::~QuantizedHNSWIndexHNSWLIB() {
    if (is_mmap_ && mmap_ptr_ != nullptr && mmap_size_ > 0) {
        munmap(mmap_ptr_, mmap_size_);
    }
}

QuantizedHNSWIndexHNSWLIB::QuantizedHNSWIndexHNSWLIB(
    std::unique_ptr<hnswlib::HierarchicalNSW<float> > index,
    int64_t next_id,
    std::unique_ptr<hnswlib::QuantizedL2Space> space
): SimilarityIndex(static_cast<int>(*static_cast<size_t *>(index->dist_func_param_))) {
    ef_search_ = static_cast<int>(index->ef_);
    ef_construction_ = static_cast<int>(index->ef_construction_);
    m_ = static_cast<int>(index->M_);
    max_elements_ = index->max_elements_;
    next_id_.store(next_id, std::memory_order_relaxed);
    index_ = std::move(index);
    space_ = std::move(space);
}

int64_t QuantizedHNSWIndexHNSWLIB::insert(
    const TensorType::VectorUInt8 &vector,
    const double scale,
    const double zero_point
) {
    const int64_t id = next_id_.fetch_add(1, std::memory_order_relaxed);
    {
        std::shared_lock resize_lock(resize_mutex_);    // prevent resizing while adding a point
        query_insertion_barrier_.enterInsertion();

        if (id < max_elements_) {
            addPointInternal(vector, scale, zero_point, id);
            query_insertion_barrier_.exitInsertion();
            return id;
        }
        query_insertion_barrier_.exitInsertion();
    }

    // need to resize the index
    {
        std::unique_lock resize_lock(resize_mutex_);
        if (id >= max_elements_)
            resize();
        query_insertion_barrier_.enterInsertion();
        addPointInternal(vector, scale, zero_point, id);
        query_insertion_barrier_.exitInsertion();
    }
    return id;
}

std::vector<int64_t> QuantizedHNSWIndexHNSWLIB::insertMany(
    std::vector<TensorType::VectorUInt8> &vectors,
    std::vector<double> &scales,
    std::vector<double> &zero_points
) {
    size_t n_tensors = vectors.size();
    std::vector<int64_t> ids(n_tensors);
    {
        std::shared_lock resize_lock(resize_mutex_);
        query_insertion_barrier_.enterInsertion();

        if (index_->getCurrentElementCount() + n_tensors <= max_elements_) {
            for (size_t i = 0; i < n_tensors; ++i) {
                const int64_t id = next_id_.fetch_add(1, std::memory_order_relaxed);
                ids[i] = id;
                addPointInternal(vectors[i], scales[i], zero_points[i], id);
            }
            query_insertion_barrier_.exitInsertion();
            return ids;
        }
    }
    {
        std::unique_lock resize_lock(resize_mutex_);
        size_t required = index_->getCurrentElementCount() + n_tensors;
        while (max_elements_ < required) {
            resize();
        }
    }
    {
        std::shared_lock resize_lock(resize_mutex_);
        query_insertion_barrier_.enterInsertion();

        for (size_t i = 0; i < n_tensors; ++i) {
            int64_t id = next_id_.fetch_add(1, std::memory_order_relaxed);
            ids[i] = id;
            addPointInternal(vectors[i], scales[i], zero_points[i], id);
        }
        query_insertion_barrier_.exitInsertion();
    }
    return ids;
}

void QuantizedHNSWIndexHNSWLIB::addPointInternal(
    const TensorType::VectorUInt8 &vector,
    double scale,
    double zero_point,
    int64_t id
) {
    size_t dimension = vector.size();
    size_t data_size = sizeof(double) + sizeof(double) + dimension;

    // insert can be called from multiple threads, so we need to use thread_local
    thread_local std::vector<uint8_t> thread_insert_buffer;
    if (thread_insert_buffer.size() < data_size) {
        thread_insert_buffer.resize(data_size);
    }
    uint8_t *buffer = thread_insert_buffer.data();
    memcpy(buffer, &scale, sizeof(double));
    memcpy(buffer + sizeof(double), &zero_point, sizeof(double));
    memcpy(
        buffer + sizeof(double) + sizeof(double),
        vector.data(),
        dimension
    );
    index_->addPoint(buffer, id);
}

std::vector<int64_t> QuantizedHNSWIndexHNSWLIB::query(
    const TensorType::VectorUInt8 &vector,
    const int k,
    double scale,
    double zero_point
) const {
    size_t dimension = vector.size();
    size_t data_size = sizeof(double) + sizeof(double) + dimension;

    // query can be called from multiple threads, so we need to use thread_local
    thread_local std::vector<uint8_t> thread_query_buffer;
    if (thread_query_buffer.size() < data_size) {
        thread_query_buffer.resize(data_size);
    }

    uint8_t *buffer = thread_query_buffer.data();
    memcpy(buffer, &scale, sizeof(double));
    memcpy(buffer + sizeof(double), &zero_point, sizeof(double));
    memcpy(
        buffer + sizeof(double) + sizeof(double),
        vector.data(),
        dimension
    );

    std::priority_queue<std::pair<float, hnswlib::labeltype>> top_k;
    {
        std::shared_lock resize_lock(resize_mutex_);
        query_insertion_barrier_.enterQuery();
        top_k = index_->searchKnn(buffer, k);
        query_insertion_barrier_.exitQuery();
    }

    std::vector<int64_t> results;
    results.reserve(top_k.size());
    while (!top_k.empty()) {
        results.push_back(static_cast<int64_t>(top_k.top().second));
        top_k.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}

TensorType::VectorFloat64 QuantizedHNSWIndexHNSWLIB::retrieve(int64_t id) const {
    std::vector<uint8_t> raw_data;
    {
        std::shared_lock resize_lock(resize_mutex_);
        raw_data = index_->getDataByLabel(id);
    }

    double scale;
    double zero_point;
    memcpy(&scale, raw_data.data(), sizeof(double));
    memcpy(&zero_point, raw_data.data() + sizeof(double), sizeof(double));

    const uint8_t *qvec_ptr = raw_data.data() + 2 * sizeof(double);
    size_t qvec_size = raw_data.size() - 2 * sizeof(double);

    Eigen::Map<const TensorType::VectorUInt8> qvec(
        qvec_ptr,
        static_cast<int>(qvec_size)
    );
    TensorType::VectorFloat64 dequantized_mat;
    LinearQuantization::linearAsymmetricDequantize(
        qvec,
        scale,
        zero_point,
        dequantized_mat
    );
    return dequantized_mat;
}

TensorType::VectorFloat16 QuantizedHNSWIndexHNSWLIB::retrieveF16(int64_t id) const {
    std::vector<uint8_t> raw_data;
    {
        std::shared_lock resize_lock(resize_mutex_);
        raw_data = index_->getDataByLabel(id);
    }

    double scale;
    double zero_point;
    memcpy(&scale, raw_data.data(), sizeof(double));
    memcpy(&zero_point, raw_data.data() + sizeof(double), sizeof(double));

    const uint8_t *qvec_ptr = raw_data.data() + 2 * sizeof(double);
    size_t qvec_size = raw_data.size() - 2 * sizeof(double);

    Eigen::Map<const TensorType::VectorUInt8> qvec(
        qvec_ptr,
        static_cast<int>(qvec_size)
    );
    TensorType::VectorFloat16 dequantized_mat_f16;
    LinearQuantization::linearAsymmetricDequantizeF16(
        qvec,
        scale,
        zero_point,
        dequantized_mat_f16
    );
    return dequantized_mat_f16;
}

UINT8QuantizedTensorPacket QuantizedHNSWIndexHNSWLIB::retrieveUINT8Quantized(int64_t id) const {
    std::vector<uint8_t> raw_data;
    {
        std::shared_lock resize_lock(resize_mutex_);
        raw_data = index_->getDataByLabel(id);
    }

    double scale;
    double zero_point;
    memcpy(&scale, raw_data.data(), sizeof(double));
    memcpy(&zero_point, raw_data.data() + sizeof(double), sizeof(double));

    const uint8_t *qvec_ptr = raw_data.data() + 2 * sizeof(double);
    size_t qvec_size = raw_data.size() - 2 * sizeof(double);

    Eigen::Map<const TensorType::VectorUInt8> qvec(
        qvec_ptr,
        static_cast<int>(qvec_size)
    );
    UINT8QuantizedTensorPacket packet;
    packet.data = qvec;
    packet.scale = scale;
    packet.zero_point = zero_point;
    packet.full_quantized_bit_width = 8;
    return packet;
}

int64_t QuantizedHNSWIndexHNSWLIB::size() const {
    return static_cast<int64_t>(index_->getCurrentElementCount());
}

int64_t QuantizedHNSWIndexHNSWLIB::space() const {
    return static_cast<int64_t>(index_->indexMemorySize());
}

void QuantizedHNSWIndexHNSWLIB::serialize(std::ostream &out) const {
    out.write(reinterpret_cast<const char *>(&m_), sizeof(m_));
    out.write(reinterpret_cast<const char *>(&ef_search_), sizeof(ef_search_));
    out.write(reinterpret_cast<const char *>(&ef_construction_), sizeof(ef_construction_));
    out.write(reinterpret_cast<const char *>(&dimension_), sizeof(dimension_));
    out.write(reinterpret_cast<const char *>(&max_elements_), sizeof(max_elements_));
    out.write(reinterpret_cast<const char *>(&next_id_), sizeof(next_id_));

    index_->saveIndexToStream(out);
}

std::shared_ptr<QuantizedHNSWIndexHNSWLIB> QuantizedHNSWIndexHNSWLIB::deserialize(std::istream &in) {
    int m, ef_search, ef_construction;
    size_t max_elements;
    int64_t dimension, next_id;

    in.read(reinterpret_cast<char*>(&m), sizeof m);
    in.read(reinterpret_cast<char*>(&ef_search), sizeof ef_search);
    in.read(reinterpret_cast<char*>(&ef_construction), sizeof ef_construction);
    in.read(reinterpret_cast<char*>(&dimension), sizeof dimension);
    in.read(reinterpret_cast<char*>(&max_elements), sizeof max_elements);
    in.read(reinterpret_cast<char*>(&next_id), sizeof next_id);

    auto space = std::make_unique<hnswlib::QuantizedL2Space>(dimension);
    auto hnsw  = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), max_elements);

    hnsw->loadIndexFromStream(in, space.get(), max_elements);

    return std::make_shared<QuantizedHNSWIndexHNSWLIB>(std::move(hnsw), next_id, std::move(space));
}

std::shared_ptr<QuantizedHNSWIndexHNSWLIB> QuantizedHNSWIndexHNSWLIB::deserializeFromMMap(
    const uint8_t *data, size_t size) {
    size_t offset = 0;
    int m, ef_search, ef_construction;
    int64_t dimension, next_id;
    size_t max_elements;

    std::memcpy(&m, data + offset, sizeof(m));
    offset += sizeof(m);
    std::memcpy(&ef_search, data + offset, sizeof(ef_search));
    offset += sizeof(ef_search);
    std::memcpy(&ef_construction, data + offset, sizeof(ef_construction));
    offset += sizeof(ef_construction);
    std::memcpy(&dimension, data + offset, sizeof(dimension));
    offset += sizeof(dimension);
    std::memcpy(&max_elements, data + offset, sizeof(max_elements));
    offset += sizeof(max_elements);
    std::memcpy(&next_id, data + offset, sizeof(next_id));
    offset += sizeof(next_id);

    auto space = std::make_unique<hnswlib::QuantizedL2Space>(dimension);
    auto index = std::make_unique<hnswlib::HierarchicalNSW<float> >(space.get());
    index->loadIndexFromMemory(data + offset, size - offset, space.get());

    auto qhnsw = std::make_shared<QuantizedHNSWIndexHNSWLIB>(
        std::move(index),
        next_id,
        std::move(space)
    );

    qhnsw->ef_search_ = ef_search;
    qhnsw->ef_construction_ = ef_construction;
    qhnsw->m_ = m;
    qhnsw->dimension_ = dimension;

    qhnsw->is_mmap_ = true;
    qhnsw->mmap_ptr_ = const_cast<uint8_t *>(data);
    qhnsw->mmap_size_ = size;
    return qhnsw;
}


void QuantizedHNSWIndexHNSWLIB::resize() {
    size_t new_capacity = max_elements_ * 2;
    index_->resizeIndex(new_capacity);
    max_elements_ = new_capacity;
}
