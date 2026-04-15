/* -------------------------------------------------------------------------
 * indexengine.h
 * RocksDB-based index storage engine using C++ API
 *
 * This is a C++ implementation of an index storage engine that uses
 * RocksDB's native C++ API for better performance and features.
 * -------------------------------------------------------------------------
 */
#ifndef INDEXENGINE_H
#define INDEXENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nrindex_access/nrindex_kv.h"
#include "postgres.h"

/* Opaque pointer to C++ IndexEngine class */
typedef struct IndexEngine IndexEngine;

/* Index engine lifecycle functions */
IndexEngine* indexengine_open(void);
void indexengine_close(IndexEngine* engine);

/* Index operations */
void indexengine_put(IndexEngine* engine, NRIndexKey ikey, NRIndexValue ivalue);
NRIndexValue indexengine_get(IndexEngine* engine, NRIndexKey ikey);
void indexengine_delete(IndexEngine* engine, NRIndexKey ikey);

/* Range scan operations */
void indexengine_range_scan(IndexEngine* engine, 
                           NRIndexKey start_key, 
                           NRIndexKey end_key,
                           uint32_t* out_count, 
                           NRIndexKey** keys, 
                           NRIndexValue** values);

/* Utility functions */
bool indexengine_exists(IndexEngine* engine, NRIndexKey ikey);
void indexengine_clear_range(IndexEngine* engine, NRIndexKey start_key, NRIndexKey end_key);

/* Bulk load operations - for efficient index building (supports INT and BIGINT) */
void indexengine_bulk_load(IndexEngine* engine,
                           Oid indexOid,
                           int64_t* keys,
                           uint64_t* values,
                           int count);

/* Statistics and maintenance */
uint64_t indexengine_get_count(IndexEngine* engine, Oid indexOid);
void indexengine_compact(IndexEngine* engine);

#ifdef __cplusplus
}
#endif

#endif /* INDEXENGINE_H */