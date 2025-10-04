#ifndef HNSW_HNSWLIB_H
#define HNSW_HNSWLIB_H

#include <hnswlib/hnswlib.h>

#include "neurstore/compress/index/similarity_index.h"


class HNSWIndexHNSWLIB final : public SimilarityIndex<TensorType::VectorFloat32> {
public:
    explicit HNSWIndexHNSWLIB(
        int dimension,
        int ef_search = 100,
        int ef_construction = 200,
        int m = 32,
        int64_t next_id = 0,
        size_t max_elements = 400
    );

    HNSWIndexHNSWLIB(
        std::unique_ptr<hnswlib::HierarchicalNSW<float> > index,
        int64_t next_id
    );

    int64_t insert(const TensorType::VectorFloat32 &vector, double scale, double zero_point) override;

    std::vector<int64_t> insertMany(std::vector<TensorType::VectorFloat32> &vectors, std::vector<double> &scales,
                                    std::vector<double> &zero_points) override;

    std::vector<int64_t> query(const TensorType::VectorFloat32 &vector, int k, double scale,
                               double zero_point) const override;

    TensorType::VectorFloat64 retrieve(int64_t id) const override;

    TensorType::VectorFloat16 retrieveF16(int64_t id) const override;

    UINT8QuantizedTensorPacket retrieveUINT8Quantized(int64_t id) const override;

    int64_t size() const override;

    int64_t space() const override;

    void resize();

    void serialize(std::ostream &out) const override;

    static std::shared_ptr<HNSWIndexHNSWLIB> deserialize(const std::vector<uint8_t> &data);

private:
    std::unique_ptr<hnswlib::SpaceInterface<float> > space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float> > index_;

    int ef_search_;
    int ef_construction_;
    int m_;
    size_t max_elements_;
    int64_t next_id_;
};

#endif //HNSW_HNSWLIB_H
