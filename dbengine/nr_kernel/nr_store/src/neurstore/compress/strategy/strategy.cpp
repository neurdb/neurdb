#include "neurstore/compress/strategy/strategy.h"

#include <random>

#include "neurstore/compress/strategy/distance_strategy.h"
#include "neurstore/compress/strategy/standard_nbits_strategy.h"


bool SelectionStrategy::selectAsRepresentative(
    const TensorType::VectorFloat64 &representative_tensor,
    const TensorType::VectorFloat64 &tensor
) {
    const double probability = getProbability(representative_tensor, tensor);
    thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(gen) < probability;
}

std::shared_ptr<SelectionStrategy> SelectionStrategy::deserialize(const std::vector<uint8_t> &data) {
    if (data.empty()) {
        throw std::runtime_error("SelectionStrategy::deserialize: data is empty");
    }
    int strategy_type;
    std::memcpy(&strategy_type, data.data(), sizeof(int));
    const uint8_t *ptr = data.data() + sizeof(int);

    if (strategy_type == 1) {
        return DistanceStrategy::deserialize({ptr, ptr + data.size() - sizeof(int)});
    }
    if (strategy_type == 2) {
        return StandardNBitsStrategy::deserialize({ptr, ptr + data.size() - sizeof(int)});
    }
    throw std::runtime_error("SelectionStrategy::deserialize: Unknown strategy type");
}

std::vector<uint8_t> SelectionStrategy::serialize(const std::shared_ptr<SelectionStrategy> &strategy) {
    if (strategy == nullptr) {
        throw std::runtime_error("SelectionStrategy::serialize: strategy is nullptr");
    }
    std::vector<uint8_t> data;
    int strategy_type;
    if (dynamic_cast<DistanceStrategy *>(strategy.get())) {
        strategy_type = 1;
        data = strategy->serialize();
    } else if (dynamic_cast<StandardNBitsStrategy *>(strategy.get())) {
        strategy_type = 2;
        data = strategy->serialize();
    } else {
        throw std::runtime_error("SelectionStrategy::serialize: Unknown strategy type");
    }
    std::vector<uint8_t> result(sizeof(int) + data.size());
    std::memcpy(result.data(), &strategy_type, sizeof(int));
    std::memcpy(result.data() + sizeof(int), data.data(), data.size());
    return result;
}
