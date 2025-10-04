#ifndef KMEANS_ROUTER_H
#define KMEANS_ROUTER_H

#include "neurstore/compress/index/router/router.h"


template<typename T>
class KMeansRouter final : public Router<T> {
public:
    KMeansRouter(int dimension, int nindex, int bit_width = 32)
        : Router<T>(dimension, nindex, bit_width) {
    }

    int route(const T &vector) override;

    void add(const T &vector) override;

    void update(int index_id, const T &vector, int num_of_vectors) override;

    void serialize(std::ostream &out) const override;

    static std::shared_ptr<KMeansRouter> deserialize(const uint8_t *data, size_t size);

private:
    std::vector<std::vector<uint8_t>> centroids_;
};

typedef KMeansRouter<TensorType::VectorFloat32> VectorFloat32KMeansRouter;

#endif //KMEANS_ROUTER_H
