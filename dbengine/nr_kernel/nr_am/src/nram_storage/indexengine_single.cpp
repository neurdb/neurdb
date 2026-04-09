/* -------------------------------------------------------------------------
 * indexengine.cpp
 * Learned index storage engine using SELIX
 * -------------------------------------------------------------------------
 */

extern "C" {
#include "indexengine.h"
#include "nrindex_access/nrindex_kv.h"
#include "utils/memutils.h"
#include "postgres.h"
#include "storage/itemptr.h"
}

#include "ALEX/src/core/alex.h"
#include "ALEX/alex_config.h"
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>

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
 * ALEX Index Engine
 * -------------------------------------------------------------------------
 */
class ALEXIndexEngine {
private:
    std::map<Oid, alex::Alex<int64_t, uint64_t>*> indexes;
    alex::ALEXConfig config;

public:
    ALEXIndexEngine() {
        // 从配置文件加载
        const char* path = "/code/neurdb-dev/dbengine/nr_kernel/nr_am/src/nram_storage/ALEX/alex_config.conf";

        if (!alex::ALEXConfigManager::instance().load_from_file(path)) {
            elog(ERROR, "ALEX IndexEngine: failed to load config from %s", path);
        }

        config = alex::ALEXConfigManager::instance().config();
        elog(LOG, "ALEX IndexEngine: loaded config from %s", path);

        // 应用成本模型权重
        alex::kExpSearchIterationsWeight = config.exp_search_iterations_weight;
        alex::kShiftsWeight = config.shifts_weight;
        alex::kNodeLookupsWeight = config.node_lookups_weight;
        alex::kModelSizeWeight = config.model_size_weight;

        elog(LOG, "ALEX config: node=%dMB, density=(%.2f,%.2f,%.2f)",
             config.max_node_size >> 20,
             config.init_density, config.max_density, config.min_density);
    }

    ~ALEXIndexEngine() {
        for (auto& pair : indexes) {
            delete pair.second;
        }
        indexes.clear();
    }

    alex::Alex<int64_t, uint64_t>* getIndex(Oid indexOid) {
        auto it = indexes.find(indexOid);
        if (it == indexes.end()) {
            alex::Alex<int64_t, uint64_t>* idx = new alex::Alex<int64_t, uint64_t>();

            // 应用配置
            idx->set_expected_insert_frac(config.expected_insert_frac);
            idx->set_max_node_size(config.max_node_size);
            idx->set_approximate_model_computation(config.approximate_model);
            idx->set_approximate_cost_computation(config.approximate_cost);
            idx->set_density_params(config.init_density, config.max_density, config.min_density);

            indexes[indexOid] = idx;
            return idx;
        }
        return it->second;
    }

    void put(Oid indexOid, int64_t val, uint64_t tid) {
        alex::Alex<int64_t, uint64_t>* idx = getIndex(indexOid);
        int64_t key = (int64_t)encode_key_64(val);
        idx->insert(key, tid);
    }

    bool get(Oid indexOid, int64_t val, uint64_t* tid) {
        auto it = indexes.find(indexOid);
        if (it == indexes.end()) return false;

        alex::Alex<int64_t, uint64_t>* idx = it->second;
        int64_t key = (int64_t)encode_key_64(val);
        auto iter = idx->find(key);
        if (!iter.is_end()) {
            *tid = iter.payload();
            return true;
        }
        return false;
    }

    bool exists(Oid indexOid, int64_t val) {
        auto it = indexes.find(indexOid);
        if (it == indexes.end()) return false;

        alex::Alex<int64_t, uint64_t>* idx = it->second;
        int64_t key = (int64_t)encode_key_64(val);
        auto iter = idx->find(key);
        return !iter.is_end();
    }

    size_t getCount(Oid indexOid) {
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

        alex::Alex<int64_t, uint64_t>* idx = getIndex(indexOid);
        idx->bulk_load(pairs.data(), pairs.size());

        auto stats = idx->get_stats();
        elog(LOG, "ALEX: bulkLoad completed, indexOid=%u, count=%zu, data_nodes=%d, model_nodes=%d",
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
        return reinterpret_cast<IndexEngine*>(new ALEXIndexEngine());
    } catch (const std::exception& e) {
        elog(ERROR, "Failed to create IndexEngine: %s", e.what());
        return nullptr;
    }
}

void indexengine_close(IndexEngine* engine) {
    if (engine) {
        delete reinterpret_cast<ALEXIndexEngine*>(engine);
    }
}

void indexengine_put(IndexEngine* engine, NRIndexKey ikey, NRIndexValue ivalue) {
    if (!engine) return;

    try {
        ALEXIndexEngine* impl = reinterpret_cast<ALEXIndexEngine*>(engine);
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
        ALEXIndexEngine* impl = reinterpret_cast<ALEXIndexEngine*>(engine);
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
            ALEXIndexEngine* impl = reinterpret_cast<ALEXIndexEngine*>(engine);
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
        ALEXIndexEngine* impl = reinterpret_cast<ALEXIndexEngine*>(engine);
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
        ALEXIndexEngine* impl = reinterpret_cast<ALEXIndexEngine*>(engine);
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
        ALEXIndexEngine* impl = reinterpret_cast<ALEXIndexEngine*>(engine);
        impl->bulkLoad(indexOid, keys, values, count);
    } catch (const std::exception& e) {
        elog(ERROR, "IndexEngine bulk_load failed: %s", e.what());
    }
}

} // extern "C"
