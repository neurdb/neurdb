/* ------------------------------------------------------------------------
 * kvengine.h
 * NeurDB key-value store management definitions.
 *
 * ------------------------------------------------------------------------
 */

#ifndef KVENGINE_H
#define KVENGINE_H

#include "kv_access/kv.h"

/* ------------------------------------------------------------------------
 * KVEngineIterator
 *
 * KVEngineIterator is an interface for iterating over key-value pairs in a
 * KV store. It is a subcomponent of the KVEngine interface.
 * ------------------------------------------------------------------------
 */
typedef struct KVEngineIterator {
    void (*destroy)(struct KVEngineIterator *);                 /* destroy the iterator */
    void (*seek)(struct KVEngineIterator *, TKey);              /* move to the first entry with key >= given key */
    bool (*is_valid)(struct KVEngineIterator *);                /* check if the iterator is valid */
    void (*next)(struct KVEngineIterator *);                    /* move to the next entry */
    void (*get)(struct KVEngineIterator *, TKey *, TValue *);   /* get the current key and value */
} KVEngineIterator;

/* ------------------------------------------------------------------------
 * KVEngine
 *
 * KVEngine is an interface for a key-value store.
 * ------------------------------------------------------------------------
 */
typedef struct KVEngine {
    void (*destroy)(struct KVEngine *);
    KVEngineIterator *(*create_iterator)(struct KVEngine *, bool isforward);
    TValue (*get)(struct KVEngine *, TKey);
    void (*put)(struct KVEngine *, TKey, TValue);
    void (*delete)(struct KVEngine *, TKey);

    /* utility functions */
    TKey (*get_min_key)(struct KVEngine *);
    TKey (*get_max_key)(struct KVEngine *);
} KVEngine;

#endif //KVENGINE_H
