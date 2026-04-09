/* ------------------------------------------------------------------------
 * nrindex_kv.c
 * NeurDB Index key-value implementation for nrindex access method.
 * ------------------------------------------------------------------------
 */

#include "nrindex_kv.h"
#include "nram_storage/rocks_service.h"
#include "nram_storage/rocks_handler.h"
#include "nram_storage/indexengine.h"  /* 直接调用 IndexEngine */
#include "utils/datum.h"

/* ------------------------------------------------------------------------
 * Local IndexEngine instance for direct calls (eliminates IPC overhead)
 * This is used in single-Backend mode for maximum performance.
 * ------------------------------------------------------------------------
 */
static IndexEngine *local_index_engine = NULL;

static IndexEngine* get_local_index_engine(void)
{
    if (local_index_engine == NULL) {
        local_index_engine = indexengine_open();
        elog(LOG, "nrindex_kv: Created local IndexEngine instance (direct call mode)");
    }
    return local_index_engine;
}

#include "utils/lsyscache.h"
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/rel.h"
#include "utils/memutils.h"

/* ------------------------------------------------------------------------
 * Big-endian encoding helpers for sortable key serialization
 * These ensure integer keys are properly ordered when compared lexicographically
 * ------------------------------------------------------------------------
 */

static inline void encode_uint32_be(char *buf, uint32 val)
{
    buf[0] = (val >> 24) & 0xFF;
    buf[1] = (val >> 16) & 0xFF;
    buf[2] = (val >> 8) & 0xFF;
    buf[3] = val & 0xFF;
}

static inline uint32 decode_uint32_be(const char *buf)
{
    return ((uint32)(unsigned char)buf[0] << 24) |
           ((uint32)(unsigned char)buf[1] << 16) |
           ((uint32)(unsigned char)buf[2] << 8) |
           ((uint32)(unsigned char)buf[3]);
}

static inline void encode_int32_be(char *buf, int32 val)
{
    /* For signed integers, flip the sign bit to maintain sort order:
     * - Negative numbers (sign bit = 1) should come before positive (sign bit = 0)
     * - After flipping: negative becomes 0..., positive becomes 1...
     * This makes lexicographic order match numeric order */
    uint32 uval = (uint32)val ^ 0x80000000;
    encode_uint32_be(buf, uval);
}

static inline int32 decode_int32_be(const char *buf)
{
    uint32 uval = decode_uint32_be(buf);
    return (int32)(uval ^ 0x80000000);
}

static inline void encode_int64_be(char *buf, int64 val)
{
    uint64 uval = (uint64)val ^ 0x8000000000000000ULL;
    buf[0] = (uval >> 56) & 0xFF;
    buf[1] = (uval >> 48) & 0xFF;
    buf[2] = (uval >> 40) & 0xFF;
    buf[3] = (uval >> 32) & 0xFF;
    buf[4] = (uval >> 24) & 0xFF;
    buf[5] = (uval >> 16) & 0xFF;
    buf[6] = (uval >> 8) & 0xFF;
    buf[7] = uval & 0xFF;
}

static inline int64 decode_int64_be(const char *buf)
{
    uint64 uval = ((uint64)(unsigned char)buf[0] << 56) |
                  ((uint64)(unsigned char)buf[1] << 48) |
                  ((uint64)(unsigned char)buf[2] << 40) |
                  ((uint64)(unsigned char)buf[3] << 32) |
                  ((uint64)(unsigned char)buf[4] << 24) |
                  ((uint64)(unsigned char)buf[5] << 16) |
                  ((uint64)(unsigned char)buf[6] << 8) |
                  ((uint64)(unsigned char)buf[7]);
    return (int64)(uval ^ 0x8000000000000000ULL);
}

/* ------------------------------------------------------------------------
 * Index key implementation
 * ------------------------------------------------------------------------
 */

NRIndexKey
nrindex_key_create(Oid indexOid, Datum *values, bool *isnull, int nkeys, TupleDesc indexTupDesc)
{
    NRIndexKey ikey;
    Size total_size;
    char *pos;
    Size *lens;
    int i;

    elog(DEBUG1, "nrindex_key_create: indexOid=%u, nkeys=%d", indexOid, nkeys);

    /* Calculate total size needed - use fixed sizes for known types */
    total_size = offsetof(NRIndexKeyData, key_data);
    lens = palloc(sizeof(Size) * nkeys);

    for (i = 0; i < nkeys; i++) {
        if (!isnull[i]) {
            Form_pg_attribute attr = TupleDescAttr(indexTupDesc, i);
            Oid typid = attr->atttypid;

            /* Use fixed sizes for sortable encoding of common types */
            switch (typid) {
                case INT2OID:
                    lens[i] = 4;  /* Encode as int32 for simplicity */
                    break;
                case INT4OID:
                    lens[i] = 4;
                    break;
                case INT8OID:
                    lens[i] = 8;
                    break;
                default:
                    /* For other types, use PostgreSQL's estimation */
                    lens[i] = datumEstimateSpace(values[i], false, attr->attbyval, attr->attlen);
                    break;
            }
            total_size += lens[i];
            elog(DEBUG1, "  Key[%d]: type=%u, len=%zu, isnull=false", i, typid, lens[i]);
        } else {
            lens[i] = 0;
            elog(DEBUG1, "  Key[%d]: isnull=true", i);
        }
    }

    /* Allocate key structure */
    ikey = (NRIndexKey)palloc0(total_size);
    ikey->indexOid = indexOid;
    ikey->key_size = total_size - offsetof(NRIndexKeyData, key_data);

    elog(DEBUG1, "  Total key_size=%u, total_struct_size=%zu", ikey->key_size, total_size);

    /* Serialize key values using big-endian for sortable ordering */
    pos = ikey->key_data;
    for (i = 0; i < nkeys; i++) {
        if (!isnull[i]) {
            Form_pg_attribute attr = TupleDescAttr(indexTupDesc, i);
            Oid typid = attr->atttypid;

            switch (typid) {
                case INT2OID:
                    /* Encode int16 as int32 in big-endian */
                    encode_int32_be(pos, (int32)DatumGetInt16(values[i]));
                    pos += 4;
                    break;
                case INT4OID:
                    encode_int32_be(pos, DatumGetInt32(values[i]));
                    pos += 4;
                    break;
                case INT8OID:
                    encode_int64_be(pos, DatumGetInt64(values[i]));
                    pos += 8;
                    break;
                default:
                    /* For other types, use PostgreSQL's serialization */
                    datumSerialize(values[i], false, attr->attbyval, attr->attlen, &pos);
                    break;
            }
        }
    }

    pfree(lens);
    return ikey;
}

char *
nrindex_key_serialize(NRIndexKey ikey, Size *out_len)
{
    char *buf;

    *out_len = sizeof(Oid) + sizeof(uint32) + ikey->key_size;
    buf = palloc0(*out_len);

    /* Use big-endian encoding for indexOid to ensure proper sort order across indexes */
    encode_uint32_be(buf, ikey->indexOid);
    encode_uint32_be(buf + sizeof(Oid), ikey->key_size);
    /* key_data is already encoded in big-endian in nrindex_key_create */
    memcpy(buf + sizeof(Oid) + sizeof(uint32), ikey->key_data, ikey->key_size);

    return buf;
}

NRIndexKey
nrindex_key_deserialize(const char *buf, Size len)
{
    NRIndexKey ikey;
    uint32 key_size;

    if (len < sizeof(Oid) + sizeof(uint32)) {
        elog(ERROR, "nrindex_key_deserialize: buffer too small");
    }

    /* Decode big-endian values */
    key_size = decode_uint32_be(buf + sizeof(Oid));

    if (len != sizeof(Oid) + sizeof(uint32) + key_size) {
        elog(ERROR, "nrindex_key_deserialize: buffer size mismatch");
    }

    ikey = (NRIndexKey)palloc0(offsetof(NRIndexKeyData, key_data) + key_size);
    ikey->indexOid = decode_uint32_be(buf);
    ikey->key_size = key_size;
    /* key_data remains in big-endian format for comparison purposes */
    memcpy(ikey->key_data, buf + sizeof(Oid) + sizeof(uint32), key_size);

    return ikey;
}

NRIndexKey
nrindex_key_copy(NRIndexKey src)
{
    NRIndexKey dst;
    Size size = offsetof(NRIndexKeyData, key_data) + src->key_size;
    
    dst = (NRIndexKey)palloc0(size);
    memcpy(dst, src, size);
    
    return dst;
}

void
nrindex_key_free(NRIndexKey ikey)
{
    if (ikey) {
        pfree(ikey);
    }
}

int
nrindex_key_compare(NRIndexKey key1, NRIndexKey key2)
{
    if (key1->indexOid != key2->indexOid) {
        return (key1->indexOid < key2->indexOid) ? -1 : 1;
    }
    
    /* Compare serialized key data */
    int cmp = memcmp(key1->key_data, key2->key_data, 
                     Min(key1->key_size, key2->key_size));
    if (cmp != 0) {
        return cmp;
    }
    
    /* If one is longer, it's greater */
    if (key1->key_size != key2->key_size) {
        return (key1->key_size < key2->key_size) ? -1 : 1;
    }
    
    return 0;
}

bool
nrindex_key_matches_scan(NRIndexKey ikey, ScanKey scankey, int nscankeys)
{
    /* For now, implement a simple approach */
    /* TODO: Properly deserialize key values and compare with scan keys */
    return true;
}

/* ------------------------------------------------------------------------
 * Index value implementation
 * ------------------------------------------------------------------------
 */

NRIndexValue
nrindex_value_create(ItemPointer heap_tid)
{
    NRIndexValue ivalue;
    
    ivalue = (NRIndexValue)palloc0(sizeof(NRIndexValueData));
    ivalue->heap_tid = *heap_tid;
    ivalue->xact_id = GetCurrentTransactionId();
    ivalue->flags = 0;
    
    return ivalue;
}

char *
nrindex_value_serialize(NRIndexValue ivalue, Size *out_len)
{
    char *buf;
    
    *out_len = sizeof(NRIndexValueData);
    buf = palloc0(*out_len);
    memcpy(buf, ivalue, *out_len);
    
    return buf;
}

NRIndexValue
nrindex_value_deserialize(const char *buf, Size len)
{
    NRIndexValue ivalue;
    
    if (len != sizeof(NRIndexValueData)) {
        elog(ERROR, "nrindex_value_deserialize: invalid buffer size");
    }
    
    ivalue = (NRIndexValue)palloc0(sizeof(NRIndexValueData));
    memcpy(ivalue, buf, sizeof(NRIndexValueData));
    
    return ivalue;
}

NRIndexValue
nrindex_value_copy(NRIndexValue src)
{
    NRIndexValue dst;
    
    dst = (NRIndexValue)palloc0(sizeof(NRIndexValueData));
    *dst = *src;
    
    return dst;
}

void
nrindex_value_free(NRIndexValue ivalue)
{
    if (ivalue) {
        pfree(ivalue);
    }
}

/* ------------------------------------------------------------------------
 * Index-specific RocksDB operations
 * ------------------------------------------------------------------------
 */

NRIndexValue
nrindex_rocks_get(NRIndexKey ikey)
{
    /* Direct call to IndexEngine - no IPC overhead */
    return indexengine_get(get_local_index_engine(), ikey);
}

bool
nrindex_rocks_put(NRIndexKey ikey, NRIndexValue ivalue)
{
    elog(DEBUG1, "nrindex_rocks_put: indexOid=%u, key_size=%u",
         ikey->indexOid, ikey->key_size);
    elog(DEBUG1, "  heap_tid=(%u,%u), xact_id=%u, flags=%u",
         ItemPointerGetBlockNumber(&ivalue->heap_tid),
         ItemPointerGetOffsetNumber(&ivalue->heap_tid),
         ivalue->xact_id, ivalue->flags);

    /* Direct call to IndexEngine - no IPC overhead */
    indexengine_put(get_local_index_engine(), ikey, ivalue);

    elog(DEBUG1, "  IndexEngine put completed");
    return true;
}

bool
nrindex_rocks_delete(NRIndexKey ikey)
{
    /* Direct call to IndexEngine - no IPC overhead */
    indexengine_delete(get_local_index_engine(), ikey);
    return true;
}

bool
nrindex_rocks_range_scan(NRIndexKey min_key, NRIndexKey max_key,
                        NRIndexKey **keys_out, NRIndexValue **values_out,
                        int *count_out)
{
    uint32_t count = 0;

    /* Direct call to IndexEngine - no IPC overhead */
    indexengine_range_scan(get_local_index_engine(), min_key, max_key,
                          &count, keys_out, values_out);

    *count_out = (int)count;
    return true;
}

/*
 * Point lookup - optimized for equality queries (WHERE val = X)
 * Much faster than range_scan for single key lookup:
 * - No array allocation
 * - No result copying
 * - Direct return of value
 */
bool
nrindex_rocks_point_lookup(NRIndexKey key, NRIndexValue *value_out, bool *found)
{
    NRIndexValue result;

    /* Direct call to IndexEngine - no IPC overhead */
    result = indexengine_get(get_local_index_engine(), key);

    /* indexengine_get returns nullptr if not found */
    if (result != NULL) {
        *value_out = result;
        *found = true;
    } else {
        *value_out = NULL;
        *found = false;
    }

    return true;
}

void
nrindex_rocks_bulk_load(Oid indexOid, int64 *keys, uint64 *values, int count)
{
    /* Direct call to IndexEngine - no IPC overhead */
    indexengine_bulk_load(get_local_index_engine(), indexOid, keys, values, count);
}
