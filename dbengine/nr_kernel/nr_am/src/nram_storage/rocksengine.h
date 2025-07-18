/* -------------------------------------------------------------------------
 * rocksengine.h
 * RocksDB kv store management definitions.
 *
 * -------------------------------------------------------------------------
 */
#ifndef ROCKSENGINE_H
#define ROCKSENGINE_H

#include "rocksdb/c.h"
#include "nram_access/kv.h"

#define NRAM_TABLE_KEY_LENGTH (sizeof(Oid))

#define ROCKS_ENGINE_MAGIC 0xCAFEBABE

typedef struct RocksEngine {
    KVEngine engine;
    rocksdb_t *rocksdb;
    rocksdb_options_t *rocksdb_options;
    uint32_t magic;
} RocksEngine;

#define SET_ROCKS_ENGINE_MAGIC(ptr)   ((ptr)->magic = ROCKS_ENGINE_MAGIC)
#define CHECK_ROCKS_ENGINE_MAGIC(ptr) ((ptr) != NULL && (ptr)->magic == ROCKS_ENGINE_MAGIC)
#define INVALIDATE_ROCKS_ENGINE_MAGIC(ptr) ((ptr)->magic = 0xDEADBEEF)

typedef struct RocksEngineIterator {
    KVEngineIterator iterator;
    rocksdb_iterator_t *rocksdb_iterator;
    rocksdb_readoptions_t *rocksdb_readoptions;
    const rocksdb_snapshot_t *rocksdb_snapshot;
} RocksEngineIterator;

/* RocksDB engine */
RocksEngine* rocksengine_open(void);
void rocksengine_open_in_place(RocksEngine* dst);
rocksdb_options_t* rocksengine_config_options(void);
void rocksengine_destroy(KVEngine *engine);
KVEngineIterator *rocksengine_create_iterator(KVEngine *engine, bool isforward);

void rocksengine_put(KVEngine *engine, NRAMKey tkey, NRAMValue tvalue);
void rocksengine_delete(KVEngine *engine, NRAMKey tkey);
NRAMValue rocksengine_get(KVEngine *engine, NRAMKey tkey);

/* Utility functions */
NRAMKey rocksengine_get_min_key(KVEngine *engine, Oid table_id);
NRAMKey rocksengine_get_max_key(KVEngine *engine, Oid table_id);

/* RocksDB iterator */
void rocksengine_fetch_table_key(Oid table_id, char** min_key, char** max_key);
void rocksengine_iterator_destroy(KVEngine *engine, KVEngineIterator *iterator);
void rocksengine_iterator_seek(KVEngineIterator *iterator, NRAMKey tkey);
void rocksengine_iterator_seek_for_prev(KVEngineIterator *iterator, NRAMKey tkey);
bool rocksengine_iterator_is_valid(KVEngineIterator *iterator);
void rocksengine_iterator_next(KVEngineIterator *iterator);
void rocksengine_iterator_prev(KVEngineIterator *iterator);
void rocksengine_iterator_get(KVEngineIterator *iterator, NRAMKey *tkey, NRAMValue *tvalue);
void rocksengine_range_scan(KVEngine *engine, NRAMKey start_key, NRAMKey end_key, uint32_t *out_count, NRAMKey **keys, NRAMValue** values);

#endif //ROCKSENGINE_H
