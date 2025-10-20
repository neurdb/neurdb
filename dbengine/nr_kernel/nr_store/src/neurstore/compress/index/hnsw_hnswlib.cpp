#include "neurstore/compress/index/hnsw_hnswlib.h"


HNSWIndexHNSWLIB::HNSWIndexHNSWLIB(
    int dimension,
    int ef_search,
    int ef_construction,
    int m,
    int64_t next_id,
    size_t max_elements
): SimilarityIndex(dimension),
   ef_search_(ef_search),
   ef_construction_(ef_construction),
   m_(m),
   max_elements_(max_elements),
   next_id_(next_id) {
    space_ = std::make_unique<hnswlib::L2Space>(dimension);
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float> >(
        space_.get(),
        max_elements_,
        m_,
        ef_construction_
    );
    index_->setEf(ef_search_);
}

HNSWIndexHNSWLIB::HNSWIndexHNSWLIB(
    std::unique_ptr<hnswlib::HierarchicalNSW<float> > index,
    int64_t next_id
): SimilarityIndex(static_cast<int>(*static_cast<size_t *>(index->dist_func_param_))) {
    ef_search_ = static_cast<int>(index->ef_);
    ef_construction_ = static_cast<int>(index->ef_construction_);
    m_ = static_cast<int>(index->M_);
    max_elements_ = index->max_elements_;
    next_id_ = next_id;
    index_ = std::move(index);
}


int64_t HNSWIndexHNSWLIB::insert(
    const TensorType::VectorFloat32 &vector,
    const double scale,
    const double zero_point
) {
    const int64_t id = next_id_++;
    if (index_->getCurrentElementCount() + 1 > max_elements_) {
        resize();
    }
    index_->addPoint(vector.data(), id);
    return id;
}

std::vector<int64_t> HNSWIndexHNSWLIB::insertMany(
    std::vector<TensorType::VectorFloat32> &vectors,
    std::vector<double> &scales,
    std::vector<double> &zero_points
) {
    size_t n_tensors = vectors.size();
    std::vector<int64_t> ids(n_tensors);
    while (index_->getCurrentElementCount() + n_tensors > max_elements_) {
        resize();
    }
    for (size_t i = 0; i < n_tensors; ++i) {
        ids[i] = next_id_++;
        index_->addPoint(vectors[i].data(), ids[i]);
    }
    return ids;
}

std::vector<int64_t> HNSWIndexHNSWLIB::query(
    const TensorType::VectorFloat32 &vector,
    const int k,
    double scale,
    double zero_point
) const {
    auto top_k = index_->searchKnn(vector.data(), k);
    std::vector<int64_t> results;
    results.reserve(top_k.size());
    while (!top_k.empty()) {
        results.push_back(static_cast<int64_t>(top_k.top().second));
        top_k.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}

TensorType::VectorFloat64 HNSWIndexHNSWLIB::retrieve(int64_t id) const {
    std::vector<uint8_t> vector = index_->getDataByLabel(id);
    std::vector<double> float_vector(vector.size() / sizeof(double));
    std::memcpy(float_vector.data(), vector.data(), vector.size());
    return Eigen::Map<TensorType::VectorFloat64>(float_vector.data(), this->dimension_);
}

TensorType::VectorFloat16 HNSWIndexHNSWLIB::retrieveF16(int64_t id) const {
    // TODO: this is not implemented
    return TensorType::VectorFloat16();
}

UINT8QuantizedTensorPacket HNSWIndexHNSWLIB::retrieveUINT8Quantized(int64_t id) const {
    // TODO: this is not implemented
    return UINT8QuantizedTensorPacket();
}

int64_t HNSWIndexHNSWLIB::size() const {
    return static_cast<int64_t>(index_->getCurrentElementCount());
}

int64_t HNSWIndexHNSWLIB::space() const {
    const std::string temp_path = "temp_index.bin";
    index_->saveIndex(temp_path);

    std::ifstream temp_file(temp_path, std::ios::binary | std::ios::ate);
    if (!temp_file.is_open()) {
        throw std::runtime_error("HNSWIndex::space: fail to open temp file");
    }
    const auto file_size = static_cast<int64_t>(temp_file.tellg());
    temp_file.close();
    std::remove(temp_path.c_str());

    return file_size;
}

void HNSWIndexHNSWLIB::serialize(std::ostream &out) const {
    out.write(reinterpret_cast<const char *>(&m_), sizeof(m_));
    out.write(reinterpret_cast<const char *>(&ef_search_), sizeof(ef_search_));
    out.write(reinterpret_cast<const char *>(&ef_construction_), sizeof(ef_construction_));
    out.write(reinterpret_cast<const char *>(&dimension_), sizeof(dimension_));
    out.write(reinterpret_cast<const char *>(&max_elements_), sizeof(max_elements_));
    out.write(reinterpret_cast<const char *>(&next_id_), sizeof(next_id_));

    index_->saveIndexToStream(out);
}

std::shared_ptr<HNSWIndexHNSWLIB> HNSWIndexHNSWLIB::deserialize(const std::vector<uint8_t> &data) {
    std::istringstream iss(std::string(data.begin(), data.end()), std::ios::binary);

    int m;
    int ef_search;
    int ef_construction;
    int64_t dimension;
    size_t max_elements;
    int64_t next_id;
    size_t meta_size = sizeof(m) + sizeof(ef_search) + sizeof(ef_construction) + sizeof(dimension) + sizeof(
                           max_elements) + sizeof(next_id);

    iss.read(reinterpret_cast<char *>(&m), sizeof(m));
    iss.read(reinterpret_cast<char *>(&ef_search), sizeof(ef_search));
    iss.read(reinterpret_cast<char *>(&ef_construction), sizeof(ef_construction));
    iss.read(reinterpret_cast<char *>(&dimension), sizeof(dimension));
    iss.read(reinterpret_cast<char *>(&max_elements), sizeof(max_elements));
    iss.read(reinterpret_cast<char *>(&next_id), sizeof(next_id));

    std::string leftover_str(
        reinterpret_cast<const char *>(data.data() + meta_size),
        data.size() - meta_size
    );
    std::istringstream leftover_iss(leftover_str, std::ios::binary);

    // load index_
    auto space = std::make_unique<hnswlib::L2Space>(dimension);
    auto hnsw = std::make_unique<hnswlib::HierarchicalNSW<float> >(space.get(), max_elements);
    hnsw->loadIndexFromStream(leftover_iss, space.get(), max_elements);

    return std::make_shared<HNSWIndexHNSWLIB>(std::move(hnsw), next_id);
}

void HNSWIndexHNSWLIB::resize() {
    size_t new_capacity = max_elements_ * 2;
    index_->resizeIndex(new_capacity);
    max_elements_ = new_capacity;
}
