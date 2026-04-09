/* -------------------------------------------------------------------------
 * indexengine_concurrent.cpp
 * Learned index storage engine using concurrent ALEX (alexol)
 *
 * 支持多线程并发访问的版本
 *
 * 切换方式:
 *   使用并发版本: 在 Makefile 中将 indexengine.cpp 替换为 indexengine_concurrent.cpp
 *   或者重命名文件:
 *     mv indexengine.cpp indexengine_single.cpp
 *     mv indexengine_concurrent.cpp indexengine.cpp
 * -------------------------------------------------------------------------
 */

extern "C" {
#include "indexengine.h"
#include "nrindex_access/nrindex_kv.h"
#include "utils/memutils.h"
#include "postgres.h"
#include "storage/itemptr.h"
}

// 保存并取消 PostgreSQL 的 LOG 定义，避免与 alexol 冲突
#ifdef LOG
#undef LOG
#endif

// 使用并发版本的 ALEX (alexol)
#include "ALEX_CUR/alexol/src/alex.h"

// 取消 alexol 的 LOG 定义，恢复 PostgreSQL 的 LOG (值为 15)
#ifdef LOG
#undef LOG
#endif
#define LOG 15
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <mutex>

/* -------------------------------------------------------------------------
 * Key encoding/decoding (supports both INT and BIGINT)
 * -------------------------------------------------------------------------
 */

// Encode int64 to uint64 for correct sort order (flip sign bit)
static inline uint64_t encode_key_64(int64_t val) {
    return (uint64_t)val ^ 0x8000000000000000ULL;
}

// Decode uint64 back to int64
static inline int64_t decode_key_64(uint64_t key) {
    return (int64_t)(key ^ 0x8000000000000000ULL);
}

// Extract key from NRIndexKey - supports both 4-byte (INT) and 8-byte (BIGINT)
static inline int64_t extract_int_from_key(NRIndexKey ikey) {
    const unsigned char* buf = (const unsigned char*)ikey->key_data;

    if (ikey->key_size == 8) {
        // BIGINT: 8 bytes, big-endian
        uint64_t encoded = ((uint64_t)buf[0] << 56) |
                           ((uint64_t)buf[1] << 48) |
                           ((uint64_t)buf[2] << 40) |
                           ((uint64_t)buf[3] << 32) |
                           ((uint64_t)buf[4] << 24) |
                           ((uint64_t)buf[5] << 16) |
                           ((uint64_t)buf[6] << 8) |
                           ((uint64_t)buf[7]);
        return (int64_t)(encoded ^ 0x8000000000000000ULL);
    } else if (ikey->key_size >= 4) {
        // INT: 4 bytes, big-endian
        uint32_t encoded = ((uint32_t)buf[0] << 24) |
                           ((uint32_t)buf[1] << 16) |
                           ((uint32_t)buf[2] << 8) |
                           ((uint32_t)buf[3]);
        return (int64_t)((int32_t)(encoded ^ 0x80000000));
    } else {
        elog(ERROR, "IndexEngine: key_size too small (%u)", ikey->key_size);
        return 0;
    }
}

/* -------------------------------------------------------------------------
 * Value encoding/decoding
 * -------------------------------------------------------------------------
 */

static inline uint64_t compress_heap_tid(ItemPointer tid) {
    BlockNumber blk = ItemPointerGetBlockNumber(tid);
    OffsetNumber off = ItemPointerGetOffsetNumber(tid);
    return ((uint64_t)blk << 32) | (uint64_t)off;
}

static inline void decompress_heap_tid(uint64_t compressed, ItemPointer tid) {
    BlockNumber blk = (BlockNumber)(compressed >> 32);
    OffsetNumber off = (OffsetNumber)(compressed & 0xFFFF);
    ItemPointerSet(tid, blk, off);
}

/* -------------------------------------------------------------------------
 * Concurrent ALEX Index Engine (using alexol)
 * -------------------------------------------------------------------------
 */
class ALEXConcurrentIndexEngine {
private:
    // 使用 alexol 命名空间的并发安全 ALEX
    std::map<Oid, alexol::Alex<int64_t, uint64_t>*> indexes;
    std::mutex index_map_mutex;  // 保护 indexes map 的互斥锁

    // 配置参数
    int max_node_size = 1 << 24;      // 16MB default
    int max_data_node_size = 1 << 19; // 512KB default

public:
    ALEXConcurrentIndexEngine() {
        elog(LOG, "ALEX Concurrent IndexEngine initialized (using alexol with TBB)");
        elog(LOG, "ALEX config: max_model_node=%dMB, max_data_node=%dKB",
             max_node_size >> 20, max_data_node_size >> 10);
    }

    ~ALEXConcurrentIndexEngine() {
        std::lock_guard<std::mutex> lock(index_map_mutex);
        for (auto& pair : indexes) {
            delete pair.second;
        }
        indexes.clear();
    }

    alexol::Alex<int64_t, uint64_t>* getIndex(Oid indexOid) {
        std::lock_guard<std::mutex> lock(index_map_mutex);

        auto it = indexes.find(indexOid);
        if (it == indexes.end()) {
            alexol::Alex<int64_t, uint64_t>* idx = new alexol::Alex<int64_t, uint64_t>();

            // 设置节点大小参数
            idx->set_max_model_node_size(max_node_size);
            idx->set_max_data_node_size(max_data_node_size);

            indexes[indexOid] = idx;
            return idx;
        }
        return it->second;
    }

    void put(Oid indexOid, int64_t val, uint64_t tid) {
        alexol::Alex<int64_t, uint64_t>* idx = getIndex(indexOid);
        int64_t key = (int64_t)encode_key_64(val);
        // alexol::insert 是线程安全的
        idx->insert(key, tid);
    }

    bool get(Oid indexOid, int64_t val, uint64_t* tid) {
        alexol::Alex<int64_t, uint64_t>* idx;
        {
            std::lock_guard<std::mutex> lock(index_map_mutex);
            auto it = indexes.find(indexOid);
            if (it == indexes.end()) return false;
            idx = it->second;
        }

        int64_t key = (int64_t)encode_key_64(val);
        // alexol::get_payload 是线程安全的
        return idx->get_payload(key, tid);
    }

    bool exists(Oid indexOid, int64_t val) {
        uint64_t dummy;
        return get(indexOid, val, &dummy);
    }

    size_t getCount(Oid indexOid) {
        std::lock_guard<std::mutex> lock(index_map_mutex);
        auto it = indexes.find(indexOid);
        if (it == indexes.end()) return 0;
        return it->second->get_stats().num_keys;
    }

    void bulkLoad(Oid indexOid, int64_t* keys, uint64_t* values, int count) {
        if (count <= 0) return;

        typedef std::pair<int64_t, uint64_t> KVPair;
        std::vector<KVPair> pairs;
        pairs.reserve(count);

        for (int i = 0; i < count; i++) {
            pairs.push_back(std::make_pair((int64_t)encode_key_64(keys[i]), values[i]));
        }

        std::sort(pairs.begin(), pairs.end(),
                  [](const KVPair& a, const KVPair& b) { return a.first < b.first; });

        auto last = std::unique(pairs.begin(), pairs.end(),
                                [](const KVPair& a, const KVPair& b) { return a.first == b.first; });
        pairs.erase(last, pairs.end());

        alexol::Alex<int64_t, uint64_t>* idx = getIndex(indexOid);
        idx->bulk_load(pairs.data(), pairs.size());

        auto stats = idx->get_stats();
        elog(LOG, "ALEX(concurrent): bulkLoad completed, indexOid=%u, count=%zu, data_nodes=%d, model_nodes=%d",
             indexOid, pairs.size(), stats.num_data_nodes, stats.num_model_nodes);
    }
};

/* -------------------------------------------------------------------------
 * C interface implementation
 * -------------------------------------------------------------------------
 */

extern "C" {

IndexEngine* indexengine_open(void) {
    try {
        return reinterpret_cast<IndexEngine*>(new ALEXConcurrentIndexEngine());
    } catch (const std::exception& e) {
        elog(ERROR, "Failed to create Concurrent IndexEngine: %s", e.what());
        return nullptr;
    }
}

void indexengine_close(IndexEngine* engine) {
    if (engine) {
        delete reinterpret_cast<ALEXConcurrentIndexEngine*>(engine);
    }
}

void indexengine_put(IndexEngine* engine, NRIndexKey ikey, NRIndexValue ivalue) {
    if (!engine) return;

    try {
        ALEXConcurrentIndexEngine* impl = reinterpret_cast<ALEXConcurrentIndexEngine*>(engine);
        Oid indexOid = ikey->indexOid;
        int64_t val = extract_int_from_key(ikey);
        uint64_t compressed_tid = compress_heap_tid(&ivalue->heap_tid);
        impl->put(indexOid, val, compressed_tid);
    } catch (const std::exception& e) {
        elog(ERROR, "IndexEngine put failed: %s", e.what());
    }
}

NRIndexValue indexengine_get(IndexEngine* engine, NRIndexKey ikey) {
    if (!engine) return nullptr;

    try {
        ALEXConcurrentIndexEngine* impl = reinterpret_cast<ALEXConcurrentIndexEngine*>(engine);
        Oid indexOid = ikey->indexOid;
        int64_t val = extract_int_from_key(ikey);

        uint64_t compressed_tid;
        if (impl->get(indexOid, val, &compressed_tid)) {
            NRIndexValue ivalue = (NRIndexValue)palloc0(sizeof(NRIndexValueData));
            decompress_heap_tid(compressed_tid, &ivalue->heap_tid);
            ivalue->xact_id = InvalidTransactionId;
            ivalue->flags = 0;
            return ivalue;
        }
        return nullptr;
    } catch (const std::exception& e) {
        elog(ERROR, "IndexEngine get failed: %s", e.what());
        return nullptr;
    }
}

void indexengine_delete(IndexEngine* engine, NRIndexKey ikey) {
    elog(WARNING, "IndexEngine: delete not supported");
}

void indexengine_range_scan(IndexEngine* engine,
                           NRIndexKey start_key,
                           NRIndexKey end_key,
                           uint32_t* out_count,
                           NRIndexKey** keys,
                           NRIndexValue** values) {
    *out_count = 0;
    *keys = nullptr;
    *values = nullptr;

    if (!engine || !start_key || !end_key) return;

    // Equality query check
    if (start_key->indexOid == end_key->indexOid &&
        start_key->key_size == end_key->key_size &&
        memcmp(start_key->key_data, end_key->key_data, start_key->key_size) == 0) {

        try {
            ALEXConcurrentIndexEngine* impl = reinterpret_cast<ALEXConcurrentIndexEngine*>(engine);
            Oid indexOid = start_key->indexOid;
            int64_t val = extract_int_from_key(start_key);

            uint64_t compressed_tid;
            if (impl->get(indexOid, val, &compressed_tid)) {
                *out_count = 1;
                *keys = (NRIndexKey*)palloc(sizeof(NRIndexKey));
                *values = (NRIndexValue*)palloc(sizeof(NRIndexValue));

                (*keys)[0] = nrindex_key_copy(start_key);
                (*values)[0] = (NRIndexValue)palloc0(sizeof(NRIndexValueData));
                decompress_heap_tid(compressed_tid, &(*values)[0]->heap_tid);
                (*values)[0]->xact_id = InvalidTransactionId;
                (*values)[0]->flags = 0;
            }
        } catch (const std::exception& e) {
            elog(ERROR, "IndexEngine range_scan failed: %s", e.what());
        }
    } else {
        elog(WARNING, "IndexEngine: true range scan not supported");
    }
}

bool indexengine_exists(IndexEngine* engine, NRIndexKey ikey) {
    if (!engine) return false;

    try {
        ALEXConcurrentIndexEngine* impl = reinterpret_cast<ALEXConcurrentIndexEngine*>(engine);
        return impl->exists(ikey->indexOid, extract_int_from_key(ikey));
    } catch (const std::exception& e) {
        elog(ERROR, "IndexEngine exists failed: %s", e.what());
        return false;
    }
}

void indexengine_clear_range(IndexEngine* engine, NRIndexKey start_key, NRIndexKey end_key) {
    elog(WARNING, "IndexEngine: clear_range not supported");
}

uint64_t indexengine_get_count(IndexEngine* engine, Oid indexOid) {
    if (!engine) return 0;

    try {
        ALEXConcurrentIndexEngine* impl = reinterpret_cast<ALEXConcurrentIndexEngine*>(engine);
        return impl->getCount(indexOid);
    } catch (const std::exception& e) {
        elog(ERROR, "IndexEngine get_count failed: %s", e.what());
        return 0;
    }
}

void indexengine_compact(IndexEngine* engine) {
    // No-op for in-memory indexes
}

void indexengine_bulk_load(IndexEngine* engine,
                           Oid indexOid,
                           int64_t* keys,
                           uint64_t* values,
                           int count) {
    if (!engine) return;

    try {
        ALEXConcurrentIndexEngine* impl = reinterpret_cast<ALEXConcurrentIndexEngine*>(engine);
        impl->bulkLoad(indexOid, keys, values, count);
    } catch (const std::exception& e) {
        elog(ERROR, "IndexEngine bulk_load failed: %s", e.what());
    }
}

} // extern "C"
