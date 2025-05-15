/* ------------------------------------------------------------------------
 * kv.h
 * NeurDB key-value struct definitions.
 *
 * The structure of the key-value store is as follows:
 * TKey -> TValue
 * TValue - [TValueFieldData TValueFieldData ... ]
 *
 * ------------------------------------------------------------------------
 */

#ifndef KV_H
#define KV_H

#include "postgres.h"
#include "access/relscan.h"
#include "executor/tuptable.h"


/* TODO: move Macros to guc.c after testing */
#define MAX_KV_CACHE_SIZE 200
#define ROCKSDB_PATH "pg_rocksdb"

/* ------------------------------------------------------------------------
 * TKeyData
 *
 * T(uple)KeyData is used to represent the key of a key-value tuple.
 * i.e. the primary key of a relation.
 * ------------------------------------------------------------------------
 */
typedef struct TKeyData {
    Oid rel_id;
    Size length;
    char *pk_serialized;
} TKeyData;

typedef TKeyData *TKey;

char *tkey_serialize(TKey key, Size *length);
TKey tkey_deserialize(char *data, Size length);
TKey tkey_generate_from_slot(Relation relation, TupleTableSlot *slot);
void tkey_free(TKey key);


/* ------------------------------------------------------------------------
 * TValueData
 *
 * T(uple)ValueData is used to represent the value of a key-value tuple.
 * ------------------------------------------------------------------------
 */
typedef struct TValueFieldData {
    int16 att_num;      /* sequence number of attribute in the tuple, starting at 0 */
    Oid att_typeid;     /* type of the attribute */
    bool is_null;       /* is the attribute NULL? */
    Size length;        /* length of the data */
    char *data;         /* variable data area */
} TValueFieldData;

typedef struct TValueData {
    int16 nfields;              /* number of fields */
    TValueFieldData *fields;    /* array of fields */
} TValueData;

typedef TValueData *TValue;

Datum tvalue_get_field(TValue tvalue, int16 att_num);
char *tvalue_serialize(TValue value, Size *length);
TValue tvalue_deserialize(char *data, Size length);
TValue tvalue_generate_from_slot(Relation relation, TupleTableSlot *slot);
void tvalue_free(TValue value);

/*
 * ----------------------------------------------------------------
 * KVScanDescData
 *
 * Scan description data for a KV scan.
 * ----------------------------------------------------------------
 */
typedef struct KVScanDescData {
    TableScanDescData rs_base;
    int rs_ckv;                                /* current kv pair index in the cache */
    int rs_nkv;                                /* number of kv pairs in the cache */
    TKey start_key;                            /* start key of the scan */
    TKey next_key;                             /* next key of the scan */
    TKey end_key;                              /* end key of the scan */
    TKey cached_keys[MAX_KV_CACHE_SIZE];       /* cached keys */
    TValue cached_values[MAX_KV_CACHE_SIZE];   /* cached values */
} KVScanDescData;

typedef KVScanDescData *KVScanDesc;

#endif //KV_H
