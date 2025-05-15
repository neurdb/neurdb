/* ------------------------------------------------------------------------
 * rocksengine.c
 * RocksDB kv store management routines.
 *
 * ------------------------------------------------------------------------
 */

#include "rocksdb/c.h"
#include "kv_storage/rocksengine.h"


/* ------------------------------------------------------------------------
 * rocksengine_config_options
 * Create rocksdb initialization options.
 * ------------------------------------------------------------------------
 */
static rocksdb_options_t *
rocksengine_config_options(void) {
    rocksdb_options_t *rocksdb_options = rocksdb_options_create();
    // add configuration options here
    rocksdb_options_set_create_if_missing(rocksdb_options, 1);
    return rocksdb_options;
}

/* ------------------------------------------------------------------------
 * rocksengine_open
 * Open a RocksDB instance.
 * ------------------------------------------------------------------------
 */
RocksEngine *
rocksengine_open() {
    rocksdb_options_t *rocksdb_options = rocksengine_config_options();
    char *error = NULL;
    rocksdb_t *rocksdb = rocksdb_open(rocksdb_options, ROCKSDB_PATH, &error);
    if (error != NULL) {
        ereport(ERROR, (errmsg("RocksDB: open operation failed, %s", error)));
    }
    RocksEngine *rocks_engine = malloc(sizeof(*rocks_engine));
    rocks_engine->rocksdb = rocksdb;
    rocks_engine->rocksdb_options = rocksdb_options;

    /* Initialize the KVEngine */
    rocks_engine->engine.create_iterator = rocksengine_create_iterator;
    rocks_engine->engine.get = rocksengine_get;
    rocks_engine->engine.put = rocksengine_put;
    rocks_engine->engine.delete = rocksengine_delete;
    rocks_engine->engine.destroy = rocksengine_destroy;
    rocks_engine->engine.get_min_key = rocksengine_get_min_key;
    rocks_engine->engine.get_max_key = rocksengine_get_max_key;
    return rocks_engine;
}

/* ------------------------------------------------------------------------
 * kvengine_destroy
 * Destroy the RocksDB engine.
 * ------------------------------------------------------------------------
 */
void
rocksengine_destroy(KVEngine *engine) {
    RocksEngine *rocks_engine = (RocksEngine *) engine;
    rocksdb_options_destroy(rocks_engine->rocksdb_options);
    rocksdb_close(rocks_engine->rocksdb);
    free(rocks_engine);
    rocks_engine = NULL;
}

/* ------------------------------------------------------------------------
 * rocksengine_create_iterator
 * Create a RocksDB engine iterator.
 * ------------------------------------------------------------------------
 */
KVEngineIterator *
rocksengine_create_iterator(KVEngine *engine, bool isforward) {
    RocksEngine *rocks_engine = (RocksEngine *) engine;
    RocksEngineIterator *rocks_it = malloc(sizeof(*rocks_it));
    rocks_it->rocksdb_readoptions = rocksdb_readoptions_create();
    rocks_it->rocksdb_iterator = rocksdb_create_iterator(rocks_engine->rocksdb, rocks_it->rocksdb_readoptions);

    /* Initialize the KVEngineIterator */
    KVEngineIterator *kv_it = (KVEngineIterator *) rocks_it;
    kv_it->destroy = rocksengine_iterator_destroy;
    kv_it->is_valid = rocksengine_iterator_is_valid;
    kv_it->get = rocksengine_iterator_get;
    if (isforward) {
        kv_it->next = rocksengine_iterator_next;
        kv_it->seek = rocksengine_iterator_seek;
    } else {
        kv_it->next = rocksengine_iterator_prev;
        kv_it->seek = rocksengine_iterator_seek_for_prev;
    }
    return kv_it;
}

/* ------------------------------------------------------------------------
 * rocksengine_get
 * Get the value of the given key from the RocksDB engine.
 * ------------------------------------------------------------------------
 */
TValue
rocksengine_get(KVEngine *engine, TKey tkey) {
    RocksEngine *rocks_engine = (RocksEngine *) engine;
    rocksdb_readoptions_t *rocksdb_readoptions = rocksdb_readoptions_create();
    Size value_lenth;
    Size key_length;
    char *key = tkey_serialize(tkey, &key_length);
    char *error = NULL;
    char *value = rocksdb_get(
        rocks_engine->rocksdb,
        rocksdb_readoptions,
        key,
        key_length,
        &value_lenth,
        &error
    );
    if (error != NULL) {
        ereport(ERROR, (errmsg("RocksDB: get operation failed, %s", error)));
    }
    TValue tvalue = tvalue_deserialize(value, value_lenth);
    rocksdb_readoptions_destroy(rocksdb_readoptions);
    free(value);
    return tvalue;
}

/* ------------------------------------------------------------------------
 * rocksengine_put
 * Put the key-value pair into the RocksDB engine.
 * ------------------------------------------------------------------------
 */
void
rocksengine_put(KVEngine *engine, TKey tkey, TValue tvalue) {
    RocksEngine *rocks_engine = (RocksEngine *) engine;
    rocksdb_writeoptions_t *rocksdb_writeoptions = rocksdb_writeoptions_create();
    Size serialized_length;
    char *serialized_value = tvalue_serialize(tvalue, &serialized_length);
    Size key_length;
    char *key = tkey_serialize(tkey, &key_length);
    char *error = NULL;
    rocksdb_put(
        rocks_engine->rocksdb,
        rocksdb_writeoptions,
        key,
        key_length,
        serialized_value,
        serialized_length,
        &error
    );
    if (error != NULL) {
        ereport(ERROR, (errmsg("RocksDB: put operation failed, %s", error)));
    }
    rocksdb_writeoptions_destroy(rocksdb_writeoptions);
    free(serialized_value);
}

/* ------------------------------------------------------------------------
 * rocksengine_delete
 * Delete the key-value pair from the RocksDB engine.
 * ------------------------------------------------------------------------
 */
void
rocksengine_delete(KVEngine *engine, TKey tkey) {
    RocksEngine *rocks_engine = (RocksEngine *) engine;
    rocksdb_writeoptions_t *rocksdb_writeoptions = rocksdb_writeoptions_create();
    Size key_length;
    char *key = tkey_serialize(tkey, &key_length);
    char *error = NULL;
    rocksdb_delete(
        rocks_engine->rocksdb,
        rocksdb_writeoptions,
        key,
        key_length,
        &error
    );
    if (error != NULL) {
        ereport(ERROR, (errmsg("RocksDB: delete operation failed, %s", error)));
    }
    rocksdb_writeoptions_destroy(rocksdb_writeoptions);
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_destroy
 * Destroy the RocksDB engine iterator.
 * ------------------------------------------------------------------------
 */
void
rocksengine_iterator_destroy(KVEngineIterator *iterator) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *) iterator;
    rocksdb_readoptions_destroy(rocks_it->rocksdb_readoptions);
    rocksdb_iter_destroy(rocks_it->rocksdb_iterator);
    free(rocks_it);
    rocks_it = NULL;
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_seek
 * Seek the RocksDB engine iterator to the given key.
 * ------------------------------------------------------------------------
 */
void
rocksengine_iterator_seek(KVEngineIterator *iterator, TKey tkey) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *) iterator;
    Size key_length;
    char *key = tkey_serialize(tkey, &key_length);
    rocksdb_iter_seek(
        rocks_it->rocksdb_iterator,
        key,
        tkey->length
    );
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_seek_for_prev
 * Seek the RocksDB engine iterator to the given key in reverse order.
 * ------------------------------------------------------------------------
 */
void
rocksengine_iterator_seek_for_prev(KVEngineIterator *iterator, TKey tkey) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *) iterator;
    Size key_length;
    char *key = tkey_serialize(tkey, &key_length);
    rocksdb_iter_seek_for_prev(
        rocks_it->rocksdb_iterator,
        key,
        tkey->length
    );
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_is_valid
 * Check if the RocksDB engine iterator is valid.
 * ------------------------------------------------------------------------
 */
bool
rocksengine_iterator_is_valid(KVEngineIterator *iterator) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *) iterator;
    return rocksdb_iter_valid(rocks_it->rocksdb_iterator);
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_next
 * Move the RocksDB engine iterator to the next key.
 * ------------------------------------------------------------------------
 */
void
rocksengine_iterator_next(KVEngineIterator *iterator) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *) iterator;
    rocksdb_iter_next(rocks_it->rocksdb_iterator);
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_prev
 * Move the RocksDB engine iterator to the previous key.
 * ------------------------------------------------------------------------
 */
void
rocksengine_iterator_prev(KVEngineIterator *iterator) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *) iterator;
    rocksdb_iter_prev(rocks_it->rocksdb_iterator);
}

/* ------------------------------------------------------------------------
 * rocksengine_iterator_get
 * Get the key-value pair from the RocksDB engine iterator.
 * ------------------------------------------------------------------------
 */
void
rocksengine_iterator_get(KVEngineIterator *iterator, TKey *tkey, TValue *tvalue) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *) iterator;
    Size key_length;
    Size value_length;
    char *key = (char *) rocksdb_iter_key(rocks_it->rocksdb_iterator, &key_length);
    char *value = (char *) rocksdb_iter_value(rocks_it->rocksdb_iterator, &value_length);
    *tkey = tkey_deserialize(key, key_length);
    *tvalue = tvalue_deserialize(value, value_length);
}

/* ------------------------------------------------------------------------
 * rocksengine_get_min_key
 * Get the minimum key from the RocksDB engine.
 * ------------------------------------------------------------------------
 */
TKey
rocksengine_get_min_key(KVEngine *engine) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *) rocksengine_create_iterator(engine, true);
    if (!rocksengine_iterator_is_valid((KVEngineIterator *) rocks_it)) {
        return NULL;
    }
    rocksdb_iter_seek_to_first(rocks_it->rocksdb_iterator);
    TKey tkey;
    TValue tvalue;
    rocksengine_iterator_get((KVEngineIterator *) rocks_it, &tkey, &tvalue);
    rocksengine_iterator_destroy((KVEngineIterator *) rocks_it);
    return tkey;
}

/* ------------------------------------------------------------------------
 * rocksengine_get_max_key
 * Get the maximum key from the RocksDB engine.
 * ------------------------------------------------------------------------
 */
TKey
rocksengine_get_max_key(KVEngine *engine) {
    RocksEngineIterator *rocks_it = (RocksEngineIterator *) rocksengine_create_iterator(engine, false);
    if (!rocksengine_iterator_is_valid((KVEngineIterator *) rocks_it)) {
        return NULL;
    }
    rocksdb_iter_seek_to_last(rocks_it->rocksdb_iterator);
    TKey tkey;
    TValue tvalue;
    rocksengine_iterator_get((KVEngineIterator *) rocks_it, &tkey, &tvalue);
    rocksengine_iterator_destroy((KVEngineIterator *) rocks_it);
    return tkey;
}
