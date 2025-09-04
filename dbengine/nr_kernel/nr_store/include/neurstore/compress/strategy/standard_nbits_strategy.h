#ifndef STANDARD_NBITS_STRATEGY_H
#define STANDARD_NBITS_STRATEGY_H

#include "neurstore/compress/strategy/strategy.h"

/**
 * Decide the probability of selecting a tensor as a representative tensor based on
 * the number of bits needed to DeltaQuant store the tensor with a given tolerance
 */
class StandardNBitsStrategy final : public SelectionStrategy {
public:
    StandardNBitsStrategy(double tolerance, int nbits, double alpha = 1.0);

    double getProbability(
        const TensorType::VectorFloat64 &representative_tensor,
        const TensorType::VectorFloat64 &tensor
    ) override;

    std::vector<uint8_t> serialize() const override;

    static std::shared_ptr<StandardNBitsStrategy> deserialize(const std::vector<uint8_t> &data);

private:
    int standard_nbits_;
    double alpha_;
};

#endif //STANDARD_NBITS_STRATEGY_H
