#ifndef DISTANCE_STRATEGY_H
#define DISTANCE_STRATEGY_H

#include "neurstore/compress/strategy/strategy.h"

/**
 * Decide the probability of selecting a tensor as a representative tensor
 * based on the distance between the tensor and the representative tensor
 */
class DistanceStrategy final : public SelectionStrategy {
public:
    DistanceStrategy(double tolerance, int nbits, const std::vector<int64_t> &shape);

    double getProbability(
        const TensorType::VectorFloat64 &representative_tensor,
        const TensorType::VectorFloat64 &tensor
    ) override;

    std::vector<uint8_t> serialize() const override;

    static std::shared_ptr<DistanceStrategy> deserialize(const std::vector<uint8_t> &data);

private:
    double max_distance_;
};

#endif //DISTANCE_STRATEGY_H
