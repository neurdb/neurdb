/* ------------------------------------------------------------------------
 * nrindex_kv.h
 * NeurDB Index key-value struct definitions for nrindex access method.
 *
 * The structure of the index key-value store is as follows:
 * NRIndexKey -> NRIndexValue
 * NRIndexKey - [indexOid][serialized_index_key_values]
 * NRIndexValue - [ItemPointer][transaction_info]
 *
 * ------------------------------------------------------------------------
 */

#ifndef NRINDEX_KV_H
#define NRINDEX_KV_H

#include "postgres.h"
#include "access/relscan.h"
#include "access/htup_details.h"
#include "access/skey.h"
#include "executor/tuptable.h"
#include "access/xact.h"
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "miscadmin.h"

/* ------------------------------------------------------------------------
 * NRIndexKey - Index key structure
 * ------------------------------------------------------------------------
 */
typedef struct NRIndexKeyData {
    Oid indexOid;              /* Index relation OID */
    uint32 key_size;           /* Size of serialized key data */
    char key_data[FLEXIBLE_ARRAY_MEMBER];  /* Serialized index key values */
} NRIndexKeyData;

typedef NRIndexKeyData *NRIndexKey;

/* ------------------------------------------------------------------------
 * NRIndexValue - Index value structure
 * ------------------------------------------------------------------------
 */
typedef struct NRIndexValueData {
    ItemPointerData heap_tid;  /* Reference to heap tuple */
    TransactionId xact_id;     /* Transaction that created this index entry */
    uint16 flags;              /* Flags for index entry state */
} NRIndexValueData;

#define NRINDEXF_PRIVATE  0x0001  /* Entry is private to current transaction */
#define NRINDEXF_DELETED  0x0002  /* Entry is marked for deletion */
#define NRIndexValueIsPrivate(v)  ((v)->flags & NRINDEXF_PRIVATE)
#define NRIndexValueIsDeleted(v)  ((v)->flags & NRINDEXF_DELETED)

typedef NRIndexValueData *NRIndexValue;

/* ------------------------------------------------------------------------
 * Index key serialization/deserialization functions
 * ------------------------------------------------------------------------
 */

/* Create an index key from index relation and key values */
extern NRIndexKey nrindex_key_create(Oid indexOid, Datum *values, bool *isnull,
                                     int nkeys, TupleDesc indexTupDesc);

/* Serialize index key to byte array for RocksDB storage */
extern char *nrindex_key_serialize(NRIndexKey ikey, Size *out_len);

/* Deserialize index key from byte array */
extern NRIndexKey nrindex_key_deserialize(const char *buf, Size len);

/* Copy an index key */
extern NRIndexKey nrindex_key_copy(NRIndexKey src);

/* Free an index key */
extern void nrindex_key_free(NRIndexKey ikey);

/* Compare two index keys for ordering */
extern int nrindex_key_compare(NRIndexKey key1, NRIndexKey key2);

/* Check if index key matches scan key conditions */
extern bool nrindex_key_matches_scan(NRIndexKey ikey, ScanKey scankey, int nscankeys);

/* ------------------------------------------------------------------------
 * Index value serialization/deserialization functions
 * ------------------------------------------------------------------------
 */

/* Create an index value from heap tuple reference */
extern NRIndexValue nrindex_value_create(ItemPointer heap_tid);

/* Serialize index value to byte array for RocksDB storage */
extern char *nrindex_value_serialize(NRIndexValue ivalue, Size *out_len);

/* Deserialize index value from byte array */
extern NRIndexValue nrindex_value_deserialize(const char *buf, Size len);

/* Copy an index value */
extern NRIndexValue nrindex_value_copy(NRIndexValue src);

/* Free an index value */
extern void nrindex_value_free(NRIndexValue ivalue);

/* ------------------------------------------------------------------------
 * Index-specific RocksDB operations
 * ------------------------------------------------------------------------
 */

/* Get index value by index key */
extern NRIndexValue nrindex_rocks_get(NRIndexKey ikey);

/* Put index key-value pair */
extern bool nrindex_rocks_put(NRIndexKey ikey, NRIndexValue ivalue);

/* Delete index entry by key */
extern bool nrindex_rocks_delete(NRIndexKey ikey);

/* Range scan over index keys */
extern bool nrindex_rocks_range_scan(NRIndexKey min_key, NRIndexKey max_key,
                                    NRIndexKey **keys_out, NRIndexValue **values_out,
                                    int *count_out);

/* Point lookup - optimized for equality queries (WHERE val = X) */
extern bool nrindex_rocks_point_lookup(NRIndexKey key, NRIndexValue *value_out, bool *found);

/* Bulk load for efficient index building (supports INT and BIGINT) */
extern void nrindex_rocks_bulk_load(Oid indexOid, int64 *keys, uint64 *values, int count);

/* ------------------------------------------------------------------------
 * Index scan descriptor for nrindex
 * ------------------------------------------------------------------------
 */
typedef struct NRIndexScanDescData {
    IndexScanDescData xs_base;     /* Base scan descriptor */
    NRIndexKey min_key;            /* Minimum key for range scan */
    NRIndexKey max_key;            /* Maximum key for range scan */
    NRIndexKey *results_key;       /* Array of result keys */
    NRIndexValue *results;         /* Array of result values */
    int result_count;              /* Number of results */
    int cursor;                    /* Current position in results */
    bool is_range_scan;            /* Whether this is a range scan */
    bool is_null_scan;             /* Whether this is scanning for NULLs */
    /* Point query optimization fields */
    bool is_point_query;           /* Whether this is an equality query */
    NRIndexKey point_key;          /* Key for point lookup */
    NRIndexValue point_result;     /* Result of point lookup */
    bool point_found;              /* Whether point lookup found a result */
    bool point_returned;           /* Whether point result has been returned */
} NRIndexScanDescData;

typedef NRIndexScanDescData *NRIndexScanDesc;

#endif /* NRINDEX_KV_H */
