#include"./src/alex.h"
#include"./indexInterface.h"

int counter = 0;

template<class KEY_TYPE, class PAYLOAD_TYPE>
class alexolInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    void init(Param *param = nullptr) {}

    void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

    bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

    bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool remove(KEY_TYPE key, Param *param = nullptr);

    size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                Param *param = nullptr);

    long long memory_consumption() { return index.model_size() + index.data_size(); }

private:
    alexol::Alex <KEY_TYPE, PAYLOAD_TYPE, alexol::AlexCompare, std::allocator<
            std::pair < KEY_TYPE, PAYLOAD_TYPE>>, false>
    index;

};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                                                        Param *param) {
    index.set_max_model_node_size(
            1 << 24); // (22,15,66M) (22,16,89M) (22,17,91M) (22,18,81M) (23,17,92M) (21,17,92M) (20,17,92M)
    index.set_max_data_node_size(1 << 19);
    std::cout << "start alex bulkload" << std::endl;
    index.bulk_load(key_value, (int) num);
    std::cout << "end alex bulkload" << std::endl;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
    auto ret = index.get_payload(key, &val);
    return ret;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    return index.insert(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    return index.update(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
    int num = index.erase(key);
    if(num) return true; else return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t alexolInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                     std::pair <KEY_TYPE, PAYLOAD_TYPE> *result,
                                                     Param *param) {
    auto scan_size = index.range_scan_by_size(key_low_bound, static_cast<uint32_t>(key_num), result);
    return scan_size;
}
