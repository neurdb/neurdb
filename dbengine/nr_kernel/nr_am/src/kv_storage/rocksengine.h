/* -------------------------------------------------------------------------
 * rocksengine.h
 * RocksDB kv store management definitions.
 *
 * -------------------------------------------------------------------------
 */
#ifndef ROCKSENGINE_H
#define ROCKSENGINE_H

#include "kv_storage/kvengine.h"
#include "rocksdb/c.h"


typedef struct RocksEngine {
    KVEngine engine;
    rocksdb_t *rocksdb;
    rocksdb_options_t *rocksdb_options;
} RocksEngine;

typedef struct RocksEngineIterator {
    KVEngineIterator iterator;
    rocksdb_iterator_t *rocksdb_iterator;
    rocksdb_readoptions_t *rocksdb_readoptions;
    // rocksdb_snapshot_t *rocksdb_snapshot;    TODO (?): add snapshot if needed
} RocksEngineIterator;

/* RocksDB engine */
rocksdb_options_t* rocksengine_config_options(void);
RocksEngine* rocksengine_open();
void rocksengine_destroy(KVEngine *engine);
KVEngineIterator *rocksengine_create_iterator(KVEngine *engine, bool isforward);

void rocksengine_put(KVEngine *engine, TKey tkey, TValue tvalue);
void rocksengine_delete(KVEngine *engine, TKey tkey);
TValue rocksengine_get(KVEngine *engine, TKey tkey);

/* Utility functions */
TKey rocksengine_get_min_key(KVEngine *engine);
TKey rocksengine_get_max_key(KVEngine *engine);

/* RocksDB iterator */
void rocksengine_iterator_destroy(KVEngineIterator *iterator);
void rocksengine_iterator_seek(KVEngineIterator *iterator, TKey tkey);
void rocksengine_iterator_seek_for_prev(KVEngineIterator *iterator, TKey tkey);
bool rocksengine_iterator_is_valid(KVEngineIterator *iterator);
void rocksengine_iterator_next(KVEngineIterator *iterator);
void rocksengine_iterator_prev(KVEngineIterator *iterator);
void rocksengine_iterator_get(KVEngineIterator *iterator, TKey *tkey, TValue *tvalue);

#endif //ROCKSENGINE_H
