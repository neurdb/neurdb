/* ------------------------------------------------------------------------
 * rocksengine.c
 * RocksDB kv store management routines.
 *
 * ------------------------------------------------------------------------
 */

#include "rocksdb/c.h"
#include "nram_storage/rocksengine.h"
#include "utils/memutils.h"

rocksdb_options_t *rocksengine_config_options(void) {
    rocksdb_options_t *rocksdb_options = rocksdb_options_create();
    rocksdb_options_set_create_if_missing(rocksdb_options, 1);
    return rocksdb_options;
}


/* ------------------------------------------------------------------------
 * rocksengine_fetch_table_key
 * The table range is [min_key, max_key).
 * ------------------------------------------------------------------------
 */
void rocksengine_fetch_table_key(Oid table_id, char** min_key, char** max_key) {
    *min_key = palloc(NRAM_TABLE_KEY_LENGTH);
    memcpy(*min_key, &table_id, NRAM_TABLE_KEY_LENGTH);
    *max_key = palloc(NRAM_TABLE_KEY_LENGTH);
    table_id += 1;
    memcpy(*max_key, &table_id, NRAM_TABLE_KEY_LENGTH);
}

/* ------------------------------------------------------------------------
 * rocksengine_open
 * Open a RocksDB instance.
 * ------------------------------------------------------------------------
 */
RocksEngine *rocksengine_open(void) {
    char *error = NULL;
    RocksEngine *rocks_engine = NULL;
    rocksdb_options_t *rocksdb_options = rocksengine_config_options();
    rocksdb_t *rocksdb = NULL;
    NRAM_INFO();

    /* First attempt */
    rocksdb = rocksdb_open(rocksdb_options, ROCKSDB_PATH, &error);

    /* Check if LOCK file is stale and try recovery */
    if (error && strstr(error, "lock hold by current process")) {
        NRAM_TEST_INFO("need manual restart");
        elog(ERROR, "RocksDB unrecoverable lock detected, manual restart needed");
    }

    /* Final check */
    if (error != NULL) {
        ereport(ERROR,
                (errmsg("RocksDB: open operation failed, %s", error)));
    }

    /* Success: initialize RocksEngine */
    rocks_engine = palloc(sizeof(RocksEngine));
    rocks_engine->rocksdb = rocksdb;
    rocks_engine->rocksdb_options = rocksdb_options;

    /* Initialize KVEngine interface */
    rocks_engine->engine.create_iterator = rocksengine_create_iterator;
    rocks_engine->engine.get = rocksengine_get;
    rocks_engine->engine.put = rocksengine_put;
    rocks_engine->engine.delete = rocksengine_delete;
    rocks_engine->engine.destroy = rocksengine_destroy;
    rocks_engine->engine.get_min_key = rocksengine_get_min_key;
    rocks_engine->engine.get_max_key = rocksengine_get_max_key;
    SET_ROCKS_ENGINE_MAGIC(rocks_engine);

    return rocks_engine;
}

void rocksengine_open_in_place(RocksEngine* dst) {
    char *error = NULL;
    NRAM_INFO();
    // memset(dst, 0, sizeof(RocksEngine));
    dst->magic = 0;
    dst->rocksdb_options = rocksengine_config_options();
    /* First attempt */
    dst->rocksdb = rocksdb_open(dst->rocksdb_options, ROCKSDB_PATH, &error);
    /* Check if LOCK file is stale and try recovery */
    if (error && strstr(error, "lock hold by current process")) {
        NRAM_TEST_INFO("need manual restart");
        elog(ERROR, "RocksDB unrecoverable lock detected, manual restart needed");
    }

    /* Final check */
    if (error != NULL) {
        ereport(ERROR,
                (errmsg("RocksDB: open operation failed, %s", error)));
    }

    /* Initialize KVEngine interface */
    dst->engine.create_iterator = rocksengine_create_iterator;
    dst->engine.get = rocksengine_get;
    dst->engine.put = rocksengine_put;
    dst->engine.delete = rocksengine_delete;
    dst->engine.destroy = rocksengine_destroy;
    dst->engine.get_min_key = rocksengine_get_min_key;
    dst->engine.get_max_key = rocksengine_get_max_key;

    /* Success: initialize RocksEngine */
    SET_ROCKS_ENGINE_MAGIC(dst);
}


/* ------------------------------------------------------------------------
 * kvengine_destroy
 * Destroy the RocksDB engine.
 * ------------------------------------------------------------------------
 */
void rocksengine_destroy(KVEngine *engine) {
    RocksEngine *rocks_engine = (RocksEngine *)engine;
    rocksdb_options_destroy(rocks_engine->rocksdb_options);
    rocksdb_close(rocks_engine->rocksdb);
    pfree(rocks_engine);
    rocks_engine = NULL;
}

/* ------------------------------------------------------------------------
 * rocksengine_create_iterator
 * Create a RocksDB engine iterator.
 * ------------------------------------------------------------------------
 */
KVEngineIterator *rocksengine_create_iterator(KVEngine *engine,
                                              bool isforward) {
    RocksEngine *rocks_engine = (RocksEngine *)engine;
    RocksEngineIterator *rocks_it = palloc0(sizeof(RocksEngineIterator));
    KVEngineIterator *kv_it;

    if (rocks_engine->rocksdb == NULL)
        elog(ERROR, "[NRAM] rocks_engine->rocksdb is NULL!");

    rocks_it->rocksdb_readoptions = rocksdb_readoptions_create();
    rocks_it->rocksdb_snapshot = rocksdb_create_snapshot(rocks_engine->rocksdb);
    rocksdb_readoptions_set_snapshot(rocks_it->rocksdb_readoptions, 
        rocks_it->rocksdb_snapshot);
    rocks_it->rocksdb_iterator = rocksdb_create_iterator(rocks_engine->rocksdb, 
        rocks_it->rocksdb_readoptions);
    if (rocks_it->rocksdb_iterator == NULL) {
        rocksdb_readoptions_destroy(rocks_it->rocksdb_readoptions);
        pfree(rocks_it);
        elog(ERROR, "[NRAM] Failed to create RocksDB iterator.");
    }

    /* Initialize the KVEngineIterator */
    kv_it = (KVEngineIterator *)rocks_it;
    kv_it->is_valid = rocksengine_iterator_is_valid;
    kv_it->get = rocksengine_iterator_get;
    kv_it->next = isforward? rocksengine_iterator_next: rocksengine_iterator_prev;
    kv_it->seek = isforward? rocksengine_iterator_seek: rocksengine_iterator_seek_for_prev;
    return kv_it;
}

/* ------------------------------------------------------------------------
 * rocksengine_get
 * Get the value of the given key from the RocksDB engine.
 * ------------------------------------------------------------------------
 */
NRAMValue rocksengine_get(KVEngine *engine, NRAMKey tkey) {
    RocksEngine *rocks_engine = (RocksEngine *)engine;
    rocksdb_readoptions_t *rocksdb_readoptions = rocksdb_readoptions_create();
    Size value_lenth;
    Size key_length;
    NRAMValue tvalue;
    char *key = tkey_serialize(tkey, &key_length);
    char *error = NULL;
    char *value = rocksdb_get(rocks_engine->rocksdb, rocksdb_readoptions, key,
                              key_length, &value_lenth, &error);
    if (error != NULL)
        ereport(ERROR, (errmsg("RocksDB: get operation failed, %s", error)));
    tvalue = tvalue_deserialize(value, value_lenth);
    rocksdb_readoptions_destroy(rocksdb_readoptions);
    free(value);
    return tvalue;
}

/* ------------------------------------------------------------------------
 * rocksengine_put
 * Put the key-value pair into the RocksDB engine.
 * ------------------------------------------------------------------------
 */
void rocksengine_put(KVEngine *engine, NRAMKey tkey, NRAMValue tvalue) {
    RocksEngine *rocks_engine = (RocksEngine *)engine;
    rocksdb_writeoptions_t *rocksdb_writeoptions = rocksdb_writeoptions_create();
    Size serialized_length, key_length;
    char *key = tkey_serialize(tkey, &key_length);
    char *serialized_value = tvalue_serialize(tvalue, &serialized_length);
    char *error = NULL;

    rocksdb_put(rocks_engine->rocksdb, rocksdb_writeoptions, key, key_length,
                serialized_value, serialized_length, &error);

    if (error != NULL)
        ereport(ERROR, (errmsg("RocksDB: put operation failed, %s", error)));

    rocksdb_writeoptions_destroy(rocksdb_writeoptions);
    pfree(serialized_value);
    pfree(key);
}

/* ------------------------------------------------------------------------
 * rocksengine_delete
 * Delete the key-value pair from the RocksDB engine.
 * ------------------------------------------------------------------------
 */
void rocksengine_delete(KVEngine *engine, NRAMKey tkey) {
    RocksEngine *rocks_engine = (RocksEngine *)engine;
    rocksdb_writeoptions_t *rocksdb_writeoptions =
        rocksdb_writeoptions_create();
    Size key_length;
    char *key = tkey_serialize(tkey, &key_length);
    char *error = NULL;
    rocksdb_delete(rocks_engine->rocksdb, rocksdb_writeoptions, key, key_length,
                   &error);
    if (error != NULL)
        ereport(ERROR, (errmsg("RocksDB: delete operation failed, %s", error)));
    rocksdb_writeoptions_destroy(rocksdb_writeoptions);
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_destroy
 * Destroy the RocksDB engine iterator.
 * ------------------------------------------------------------------------
 */
void rocksengine_iterator_destroy(KVEngine* engine, KVEngineIterator *iterator) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *)iterator;
    RocksEngine *rocks_engine = (RocksEngine *)engine;
    rocksdb_release_snapshot(rocks_engine->rocksdb, rocks_it->rocksdb_snapshot);
    rocksdb_readoptions_destroy(rocks_it->rocksdb_readoptions);
    rocksdb_iter_destroy(rocks_it->rocksdb_iterator);
    rocks_it = NULL;
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_seek
 * Seek the RocksDB engine iterator to the given key.
 * ------------------------------------------------------------------------
 */
void rocksengine_iterator_seek(KVEngineIterator *iterator, NRAMKey tkey) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *)iterator;
    Size key_length;
    char *key = tkey_serialize(tkey, &key_length);
    rocksdb_iter_seek(rocks_it->rocksdb_iterator, key, key_length);
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_seek_for_prev
 * Seek the RocksDB engine iterator to the given key in reverse order.
 * ------------------------------------------------------------------------
 */
void rocksengine_iterator_seek_for_prev(KVEngineIterator *iterator,
                                        NRAMKey tkey) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *)iterator;
    Size key_length;
    char *key = tkey_serialize(tkey, &key_length);
    rocksdb_iter_seek_for_prev(rocks_it->rocksdb_iterator, key, key_length);
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_is_valid
 * Check if the RocksDB engine iterator is valid.
 * ------------------------------------------------------------------------
 */
bool rocksengine_iterator_is_valid(KVEngineIterator *iterator) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *)iterator;
    return rocksdb_iter_valid(rocks_it->rocksdb_iterator);
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_next
 * Move the RocksDB engine iterator to the next key.
 * ------------------------------------------------------------------------
 */
void rocksengine_iterator_next(KVEngineIterator *iterator) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *)iterator;
    rocksdb_iter_next(rocks_it->rocksdb_iterator);
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_prev
 * Move the RocksDB engine iterator to the previous key.
 * ------------------------------------------------------------------------
 */
void rocksengine_iterator_prev(KVEngineIterator *iterator) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *)iterator;
    rocksdb_iter_prev(rocks_it->rocksdb_iterator);
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_get
 * Get the key-value pair from the RocksDB engine iterator.
 * ------------------------------------------------------------------------
 */
void rocksengine_iterator_get(KVEngineIterator *iterator, NRAMKey *tkey,
                              NRAMValue *tvalue) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *)iterator;
    Size key_length;
    Size value_length;
    char *key =
        (char *)rocksdb_iter_key(rocks_it->rocksdb_iterator, &key_length);
    char *value =
        (char *)rocksdb_iter_value(rocks_it->rocksdb_iterator, &value_length);
    *tkey = tkey_deserialize(key, key_length);
    *tvalue = tvalue_deserialize(value, value_length);
}

/* ------------------------------------------------------------------------
 * rocksengine_get_min_key
 * Get the minimum key from the RocksDB engine.
 * ------------------------------------------------------------------------
 */
NRAMKey rocksengine_get_min_key(KVEngine *engine, Oid table_id) {
    RocksEngineIterator *rocks_it =
        (RocksEngineIterator *)rocksengine_create_iterator(engine, true);
    NRAMKey tkey;
    NRAMValue tvalue;
    char* min_key, *max_key;

    rocksengine_fetch_table_key(table_id, &min_key, &max_key);
    rocksdb_iter_seek(rocks_it->rocksdb_iterator, min_key, NRAM_TABLE_KEY_LENGTH);

    if (!rocksengine_iterator_is_valid((KVEngineIterator *)rocks_it)) {
        rocksengine_iterator_destroy(engine, (KVEngineIterator *)rocks_it);
        return NULL;
    }
    rocksengine_iterator_get((KVEngineIterator *)rocks_it, &tkey, &tvalue);
    rocksengine_iterator_destroy(engine, (KVEngineIterator *)rocks_it);
    return tkey;
}

/* ------------------------------------------------------------------------
 * rocksengine_get_max_key
 * Get the maximum key from the RocksDB engine.
 * ------------------------------------------------------------------------
 */
NRAMKey rocksengine_get_max_key(KVEngine *engine, Oid table_id) {
    RocksEngineIterator *rocks_it =
        (RocksEngineIterator *)rocksengine_create_iterator(engine, false);
    NRAMKey tkey;
    NRAMValue tvalue;
    char* min_key, *max_key;

    rocksengine_fetch_table_key(table_id, &min_key, &max_key);
    rocksdb_iter_seek_for_prev(rocks_it->rocksdb_iterator, max_key, NRAM_TABLE_KEY_LENGTH);

    if (!rocksengine_iterator_is_valid((KVEngineIterator *)rocks_it))
        return NULL;

    rocksengine_iterator_get((KVEngineIterator *)rocks_it, &tkey, &tvalue);
    rocksengine_iterator_destroy(engine, (KVEngineIterator *)rocks_it);
    return tkey;
}
