/* ------------------------------------------------------------------------
 * kv.h
 * NeurDB key-value struct definitions.
 *
 * The structure of the key-value store is as follows:
 * NRAMKey -> NRAMValue
 * NRAMValue - [NRAMValueFieldData NRAMValueFieldData ... ]
 *
 * ------------------------------------------------------------------------
 */

#ifndef KV_H
#define KV_H

#include "postgres.h"
#include "access/relscan.h"
#include "executor/tuptable.h"
#include "access/xact.h"

#define ROCKSDB_PATH "pg_rocksdb"
#define NRAM_TEST_INFO(fmt, ...) \
    elog(INFO, "[NRAM] [%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#define NRAM_INFO() elog(INFO, "[NRAM] calling function %s", __func__)


typedef struct NRAMKeyData {
    int16 nkeys;
    Size length;
    Oid tableOid;
    char data[];
} NRAMKeyData;

typedef NRAMKeyData *NRAMKey;

char *stringify_nram_key(NRAMKey key, TupleDesc desc, int *key_attrs);
NRAMKey nram_key_serialize_from_tuple(HeapTuple tuple, TupleDesc tupdesc, int *key_attrs, int nkeys);
void nram_key_deserialize(NRAMKey tkey, TupleDesc desc, int *key_attrs, Datum *values, bool *is_null);
char *stringify_buff(char *buf, int len);
char *tkey_serialize(NRAMKey tkey, Size *out_len);
NRAMKey tkey_deserialize(char *buf, Size len);


typedef struct NRAMValueFieldData {
    int16 attnum;       // Attribute number
    Oid type_oid;       // Type OID
    uint32 len;         // Length of data (0 if NULL)
    char data[FLEXIBLE_ARRAY_MEMBER];        // Actual data or empty if NULL
} NRAMValueFieldData;

typedef struct NRAMValueData {
    TransactionId tid;    // The transaction that has created this data version.   
    int16 nfields;
    char data[FLEXIBLE_ARRAY_MEMBER];  // Consecutive NRAMValueFieldData blocks
} NRAMValueData;

typedef NRAMValueData *NRAMValue;

NRAMValue nram_value_serialize_from_tuple(HeapTuple tuple, TupleDesc tupdesc);
HeapTuple deserialize_nram_value_to_tuple(NRAMValue val, TupleDesc tupdesc);
char *tvalue_serialize(NRAMValue tvalue, Size *out_len);
NRAMValue tvalue_deserialize(char *buf, Size len);


/* ------------------------------------------------------------------------
 * KVEngineIterator
 *
 * KVEngineIterator is an interface for iterating over key-value pairs in a
 * KV store. It is a subcomponent of the KVEngine interface.
 * ------------------------------------------------------------------------
 */
typedef struct KVEngineIterator {
    void (*destroy)(struct KVEngineIterator *);                 /* destroy the iterator */
    void (*seek)(struct KVEngineIterator *, NRAMKey);              /* move to the first entry with key >= given key */
    bool (*is_valid)(struct KVEngineIterator *);                /* check if the iterator is valid */
    void (*next)(struct KVEngineIterator *);                    /* move to the next entry */
    void (*get)(struct KVEngineIterator *, NRAMKey *, NRAMValue *);   /* get the current key and value */
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
    NRAMValue (*get)(struct KVEngine *, NRAMKey);
    void (*put)(struct KVEngine *, NRAMKey, NRAMValue);
    void (*delete)(struct KVEngine *, NRAMKey);

    /* utility functions */
    NRAMKey (*get_min_key)(struct KVEngine *, Oid table_id);
    NRAMKey (*get_max_key)(struct KVEngine *, Oid table_id);
} KVEngine;

/*
 * ----------------------------------------------------------------
 * KVScanDescData
 *
 * Scan description data for a KV scan.
 * ----------------------------------------------------------------
 */
typedef struct KVScanDescData {
    TableScanDescData rs_base;
    KVEngineIterator* engine_iterator;
    NRAMKey min_key;
    NRAMKey max_key;
} KVScanDescData;

typedef KVScanDescData *KVScanDesc;


#endif //KV_H
