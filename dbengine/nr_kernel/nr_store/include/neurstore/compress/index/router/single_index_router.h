#ifndef SINGLE_INDEX_ROUTER_H
#define SINGLE_INDEX_ROUTER_H

#include "neurstore/compress/index/router/router.h"
#include "neurstore/utils/global.h"


template<typename T>
class SingleIndexRouter final : public Router<T> {
public:
    SingleIndexRouter(int dimension,
        int max_elements_per_index,
        int nindex = 1,
        int current_index_id = 0,
        int current_element_count = 0,
        int bit_width = 32
    ) : Router<T>(dimension, nindex, bit_width) {
        max_elements_per_index_ = max_elements_per_index;
        current_index_id_ = current_index_id;
        current_element_count_ = current_element_count;
    }

    int route(const T &vector) override;

    void add(const T &vector) override;

    void update(int index_id, const T &vector, int num_of_vectors) override;

    void serialize(std::ostream &out) const override;

    static std::shared_ptr<SingleIndexRouter> deserialize(const uint8_t *data, size_t size);

private:
    mutable std::mutex mutex_;
    int max_elements_per_index_;
    int current_index_id_;
    int current_element_count_;
};

typedef SingleIndexRouter<TensorType::VectorFloat64> SingleIndex64Router;

#endif //SINGLE_INDEX_ROUTER_H
