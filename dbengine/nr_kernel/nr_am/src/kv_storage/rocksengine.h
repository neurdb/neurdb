/* -------------------------------------------------------------------------
 * rocksengine.h
 * RocksDB kv store management definitions.
 *
 * -------------------------------------------------------------------------
 */
#ifndef ROCKSENGINE_H
#define ROCKSENGINE_H

#include "rocksdb/c.h"
#include "kv_access/kv.h"

#define NRAM_TABLE_KEY_LENGTH 4

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
RocksEngine* rocksengine_open(void);
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
void rocksengine_iterator_destroy(KVEngineIterator *iterator);
void rocksengine_iterator_seek(KVEngineIterator *iterator, NRAMKey tkey);
void rocksengine_iterator_seek_for_prev(KVEngineIterator *iterator, NRAMKey tkey);
bool rocksengine_iterator_is_valid(KVEngineIterator *iterator);
void rocksengine_iterator_next(KVEngineIterator *iterator);
void rocksengine_iterator_prev(KVEngineIterator *iterator);
void rocksengine_iterator_get(KVEngineIterator *iterator, NRAMKey *tkey, NRAMValue *tvalue);

#endif //ROCKSENGINE_H
