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
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "miscadmin.h"
#include "nram_utils/config.h"

extern void nram_init(void);
extern void nram_generate_tid(ItemPointer tid);
extern uint64_t nram_decode_tid(const ItemPointer tid);
extern void nram_encode_tid(uint64_t logical_tid, ItemPointer tid);
// extern shmem_startup_hook_type prev_shmem_startup_hook;
// extern void nram_shmem_startup(void);

typedef struct NRAMKeyData {
    Oid tableOid;
    uint64_t tid;
} NRAMKeyData;

typedef NRAMKeyData *NRAMKey;

extern char *stringify_buff(char *buf, int len);
extern char *tkey_serialize(NRAMKey tkey, Size *out_len);
extern NRAMKey tkey_deserialize(const char *buf, Size len);
extern NRAMKey nram_key_from_tid(Oid tableOid, ItemPointer tid);
extern NRAMKey copy_nram_key(NRAMKey src);

typedef struct NRAMValueFieldData {
    int16 attnum;       // Attribute number
    Oid type_oid;       // Type OID
    uint32 len;         // Length of data (0 if NULL)
    // char data[FLEXIBLE_ARRAY_MEMBER];
} NRAMValueFieldData;

// NRAMValueData memory arrangement: [xact id] [flags] [nfields] [field1 data1] ...
// Note: in NRAM, we do not maintain multiple versions of a tuple.
// Thus, we do not need to maintain a full visibility map.
// We only need to check if the transaction that created/deleted this tuple has committed.
typedef struct NRAMValueData {
    TransactionId xact_id;    // The transaction that has created this data version.
    int16 flags;      // reserved for future use
    int16 nfields;
    char data[FLEXIBLE_ARRAY_MEMBER];  // Consecutive NRAMValueFieldData blocks
} NRAMValueData;

#define NRAMF_PRIVATE  0x0001
#define NRAMF_DELETED  0x0002
#define NRAMValueIsPrivate(v)  ((v)->flags & NRAMF_PRIVATE)
#define NRAMValueIsDeleted(v)  ((v)->flags & NRAMF_DELETED)

typedef NRAMValueData *NRAMValue;

extern NRAMValue nram_value_serialize_from_tuple(HeapTuple tuple, TupleDesc tupdesc);
extern HeapTuple deserialize_nram_value_to_tuple(NRAMValue val, TupleDesc tupdesc);
extern char *tvalue_serialize(NRAMValue tvalue, Size *out_len);
extern NRAMValue tvalue_deserialize(const char *buf, Size len);
extern NRAMValue copy_nram_value(NRAMValue src);

/* ------------------------------------------------------------------------
 * KVEngineIterator
 *
 * KVEngineIterator is an interface for iterating over key-value pairs in a
 * KV store. It is a subcomponent of the KVEngine interface.
 * ------------------------------------------------------------------------
 */
typedef struct KVEngineIterator {
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

extern KVEngine *GetCurrentEngine(void);

/*
 * The scan state is for maintaining state for a scan, either for a
 * SELECT or UPDATE or DELETE.
 */
typedef struct TableReadState {
    uint64 operationId;
    NRAMKey key;
    bool hasNext;  /* whether a next batch from RangeQuery or ReadBatch*/

    bool done;
    char* buf;     /* data returned by RangeQuery or ReadBatch */
    size_t bufLen; /* no next batch if it is 0 */
    char* next;    /* pointer to the next data entry during scan */

    bool execExplainOnly;
} TableReadState;

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
    NRAMValue *results;
    NRAMKey *results_key;
    uint32_t result_count;
    uint32_t cursor;
} KVScanDescData;

typedef KVScanDescData *KVScanDesc;

typedef struct IndexFetchKVData {
    IndexFetchTableData xs_base;   /* base structure */
} IndexFetchKVData;

#endif //KV_H
