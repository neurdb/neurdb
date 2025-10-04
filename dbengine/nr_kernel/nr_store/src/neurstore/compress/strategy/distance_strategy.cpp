#include "neurstore/compress/strategy/distance_strategy.h"


DistanceStrategy::DistanceStrategy(const double tolerance, const int nbits, const std::vector<int64_t> &shape)
    : SelectionStrategy(tolerance, nbits, shape) {
    const Eigen::VectorXf max_distance_vector = Eigen::VectorXf::Constant(
        this->shape_.front(),
        static_cast<float>(this->tolerance_ * std::pow(2, this->nbits_))
    );
    max_distance_ = max_distance_vector.norm();
}

double DistanceStrategy::getProbability(
    const TensorType::VectorFloat64 &representative_tensor,
    const TensorType::VectorFloat64 &tensor
) {
    const double distance = (representative_tensor - tensor).norm();
    if (distance == 0.0f) {
        return 0.0f;
    }
    double probability = std::log2(distance) / std::log2(max_distance_);
    probability = std::clamp(probability, 0.0, 1.0);
    return probability;
}

std::vector<uint8_t> DistanceStrategy::serialize() const {
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
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&max_distance_),
                reinterpret_cast<const uint8_t *>(&max_distance_) + sizeof(max_distance_));
    return data;
}

std::shared_ptr<DistanceStrategy> DistanceStrategy::deserialize(const std::vector<uint8_t> &data) {
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
    return std::make_shared<DistanceStrategy>(tolerance, nbits, shape);
}
