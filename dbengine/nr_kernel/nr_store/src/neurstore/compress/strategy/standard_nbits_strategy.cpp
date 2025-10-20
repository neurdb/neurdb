#include "neurstore/compress/strategy/standard_nbits_strategy.h"


StandardNBitsStrategy::StandardNBitsStrategy(
    const double tolerance,
    const int nbits,
    const double alpha
) : SelectionStrategy(tolerance, nbits, {}), alpha_(alpha) {
    standard_nbits_ = std::ceil(std::log2(0.16 / this->tolerance_));
}

double StandardNBitsStrategy::getProbability(
    const TensorType::VectorFloat64 &representative_tensor,
    const TensorType::VectorFloat64 &tensor
) {
    const TensorType::VectorFloat64 delta = representative_tensor - tensor;
    const double scale = delta.maxCoeff() - delta.minCoeff();

    if (scale == 0.0f) {
        return 0.0f;
    }

    const int nbits = std::ceil(std::log2(scale / this->tolerance_));
    if (nbits <= this->standard_nbits_) {
        return 0.0f;
    }
    double probability =
            1 - std::exp(
                -this->alpha_ * (
                    static_cast<double>(nbits) - static_cast<double>(this->standard_nbits_)
                )
            );
    probability = std::clamp(probability, 0.0, 1.0);
    return probability;
}

std::vector<uint8_t> StandardNBitsStrategy::serialize() const {
    std::vector<uint8_t> data;
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&this->tolerance_),
                reinterpret_cast<const uint8_t *>(&this->tolerance_) + sizeof(this->tolerance_));
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&this->nbits_),
                reinterpret_cast<const uint8_t *>(&this->nbits_) + sizeof(this->nbits_));
    const size_t shape_size = this->shape_.size();
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&shape_size),
                reinterpret_cast<const uint8_t *>(&shape_size) + sizeof(shape_size));
    for (const auto &dim: this->shape_) {
        data.insert(data.end(), reinterpret_cast<const uint8_t *>(&dim),
                    reinterpret_cast<const uint8_t *>(&dim) + sizeof(dim));
    }
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&this->alpha_),
                reinterpret_cast<const uint8_t *>(&this->alpha_) + sizeof(this->alpha_));
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&this->standard_nbits_),
                reinterpret_cast<const uint8_t *>(&this->standard_nbits_) + sizeof(this->standard_nbits_));
    return data;
}

std::shared_ptr<StandardNBitsStrategy> StandardNBitsStrategy::deserialize(const std::vector<uint8_t> &data) {
    const uint8_t *ptr = data.data();

    double tolerance;
    std::memcpy(&tolerance, ptr, sizeof(double));
    ptr += sizeof(double);

    int nbits;
    std::memcpy(&nbits, ptr, sizeof(int));
    ptr += sizeof(int);

    size_t shape_size;
    std::memcpy(&shape_size, ptr, sizeof(size_t));
    ptr += sizeof(size_t);

    std::vector<int64_t> shape;
    for (size_t i = 0; i < shape_size; ++i) {
        int64_t dim;
        std::memcpy(&dim, ptr, sizeof(int64_t));
        shape.push_back(dim);
        ptr += sizeof(int64_t);
    }

    double alpha;
    std::memcpy(&alpha, ptr, sizeof(double));
    ptr += sizeof(double);

    int standard_nbits;
    std::memcpy(&standard_nbits, ptr, sizeof(int));

    return std::make_shared<StandardNBitsStrategy>(tolerance, nbits, alpha);
}
