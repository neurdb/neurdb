#include "kv.h"
#include "utils/datum.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/timestamp.h"
#include "funcapi.h"
#include "nram_storage/rocksengine.h"
#include "miscadmin.h"


/* ------------------------------------------------------------------------
 * NRAM tid key generator APIs.
 * ------------------------------------------------------------------------
 */

static pg_atomic_uint64 local_tid_seq;

static void nram_init_tid(void) {
    pg_atomic_init_u64(&local_tid_seq, 0);
}

// The logical tid format is [auto inc counter] [process id]
static uint64_t nram_generate_logical_tid(void) {
    uint32 counter = (uint32)(pg_atomic_fetch_add_u64(&local_tid_seq, 1) & 0xFFFFFFFF);
    uint16 pid_part = (uint16)(MyProcPid & 0xFFFF);
    return ((uint64_t)counter << 16) | pid_part;
}

void nram_encode_tid(uint64_t logical_tid, ItemPointer tid) {
    BlockNumber block = (logical_tid >> 16) & 0xFFFFFFFF;
    OffsetNumber offset = logical_tid & 0xFFFF;
    Assert(offset != InvalidOffsetNumber);
    ItemPointerSet(tid, block, offset);
}

uint64_t nram_decode_tid(const ItemPointer tid) {
    return ((uint64_t)BlockIdGetBlockNumber(&tid->ip_blkid) << 16) | tid->ip_posid;
}

void nram_generate_tid(ItemPointer tid) {
    uint64_t logical_tid = nram_generate_logical_tid();
    nram_encode_tid(logical_tid, tid);
}


/* ------------------------------------------------------------------------
 * NRAM per session rocksdb engine APIs.
 * ------------------------------------------------------------------------
 */

static KVEngine *shared_engine = NULL;

void nram_init(void) {
    nram_init_tid();
}


KVEngine* GetCurrentEngine(void) {
    if (shared_engine == NULL) {
        MemoryContext oldCtx = MemoryContextSwitchTo(TopMemoryContext);
        shared_engine = (KVEngine*) rocksengine_open();
        MemoryContextSwitchTo(oldCtx);
    }
    Assert(CHECK_ROCKS_ENGINE_MAGIC((RocksEngine*)shared_engine));
    return shared_engine;
}

NRAMKey nram_key_from_tid(Oid tableOid, ItemPointer tid) {
    NRAMKey tkey = palloc0(sizeof(NRAMKeyData));
    tkey->tableOid = tableOid;
    tkey->tid = nram_decode_tid(tid);
    return tkey;
}

char *stringify_buff(char *buf, int len) {
    StringInfoData out;
    initStringInfo(&out);

    appendStringInfoString(&out, "[Hex] ");

    for (Size i = 0; i < len; i++) {
        appendStringInfo(&out, "%02X", (unsigned char)buf[i]);
    }

    return out.data;
}

NRAMValue nram_value_serialize_from_tuple(HeapTuple tuple, TupleDesc tupdesc) {
    Datum *values = palloc(sizeof(Datum) * tupdesc->natts);
    bool *isnull = palloc(sizeof(bool) * tupdesc->natts);
    NRAMValue val;
    Size total_size, *lens;
    char *pos;

    heap_deform_tuple(tuple, tupdesc, values, isnull);

    // Estimate space needed
    total_size = offsetof(NRAMValueData, data);
    lens = palloc(sizeof(Size) * tupdesc->natts);

    for (int i = 0; i < tupdesc->natts; i++) {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, i);
        lens[i] = datumEstimateSpace(values[i], isnull[i], attr->attbyval, attr->attlen);
        total_size += lens[i] + sizeof(NRAMValueFieldData);
    }

    val = (NRAMValueData *)palloc0(total_size);
    val->nfields = tupdesc->natts;
    val->xact_id = GetTopTransactionId();
    val->flags = 0;

    pos = (char*)val + offsetof(NRAMValueData, data);
    for (int i = 0; i < tupdesc->natts; i++) {
        NRAMValueFieldData *field = (NRAMValueFieldData *)pos;
        Form_pg_attribute attr = TupleDescAttr(tupdesc, i);

        field->attnum = i;
        field->type_oid = TupleDescAttr(tupdesc, i)->atttypid;
        field->len = lens[i];

        pos += sizeof(NRAMValueFieldData);
        datumSerialize(values[i], isnull[i], attr->attbyval, attr->attlen, &pos);
    }

    if (pos - (char*)val != total_size)
        elog(ERROR, "[nram_value_serialize_from_tuple]: wrong value serialization length mismatch: expected %zu: real %d",
            total_size,
            (int)(pos - (char*)val));


    return val;
}

HeapTuple deserialize_nram_value_to_tuple(NRAMValue val, TupleDesc tupdesc) {
    Datum *values = palloc0(sizeof(Datum) * val->nfields);
    bool *is_null = palloc0(sizeof(bool) * val->nfields);

    char *pos = (char*)val + offsetof(NRAMValueData, data);
    for (int i = 0; i < val->nfields; i++) {
        pos += sizeof(NRAMValueFieldData);
        values[i] = datumRestore(&pos, &is_null[i]);
    }

    return heap_form_tuple(tupdesc, values, is_null);
}

char *tvalue_serialize(NRAMValue tvalue, Size *out_len) {
    char *ptr = (char *)tvalue + offsetof(NRAMValueData, data);
    char *buf, *write_ptr;
    Size data_len = 0, total_len;

    for (int i = 0; i < tvalue->nfields; i++) {
        NRAMValueFieldData *f = (NRAMValueFieldData *)ptr;
        Size field_size = sizeof(NRAMValueFieldData) + f->len;
        data_len += field_size;
        ptr += field_size;
    }

    total_len = sizeof(TransactionId) + sizeof(int16) + sizeof(int16) + data_len;

    buf = palloc(total_len);
    write_ptr = buf;

    memcpy(write_ptr, &tvalue->xact_id, sizeof(TransactionId));
    write_ptr += sizeof(TransactionId);


    memcpy(write_ptr, &tvalue->flags, sizeof(int16));
    write_ptr += sizeof(int16);

    memcpy(write_ptr, &tvalue->nfields, sizeof(int16));
    write_ptr += sizeof(int16);

    ptr = (char *)tvalue + offsetof(NRAMValueData, data);
    for (int i = 0; i < tvalue->nfields; i++) {
        NRAMValueFieldData *f = (NRAMValueFieldData *)ptr;
        Size field_size = sizeof(NRAMValueFieldData) + f->len;
        memcpy(write_ptr, f, field_size);
        ptr += field_size;
        write_ptr += field_size;
    }

    Assert(write_ptr == buf + total_len);  // safety check
    *out_len = total_len;
    return buf;
}

NRAMValue tvalue_deserialize(const char *buf, Size len) {
    TransactionId xact_id;
    int16 nfields, flags;
    char *ptr;
    Size data_len, total_len;
    NRAMValue tvalue;

    // Read header
    memcpy(&xact_id, buf, sizeof(TransactionId));
    buf += sizeof(TransactionId);

    memcpy(&flags, buf, sizeof(int16));
    buf += sizeof(int16);

    memcpy(&nfields, buf, sizeof(int16));
    buf += sizeof(int16);

    data_len = len - sizeof(TransactionId) - sizeof(int16);
    total_len = offsetof(NRAMValueData, data) + data_len;

    tvalue = (NRAMValue)palloc(total_len);
    tvalue->xact_id = xact_id;
    tvalue->nfields = nfields;
    tvalue->flags = flags;

    ptr = (char *)tvalue + offsetof(NRAMValueData, data);
    memcpy(ptr, buf, data_len);

    return tvalue;
}

// Layout [Oid tableOid][data...]
char *tkey_serialize(NRAMKey tkey, Size *out_len) {
    char *buf, *ptr;

    *out_len = sizeof(Oid) + sizeof(uint64_t);
    buf = palloc0(*out_len);
    ptr = buf;
    memcpy(ptr, &tkey->tableOid, sizeof(Oid));
    ptr += sizeof(Oid);
    memcpy(ptr, &tkey->tid, sizeof(uint64_t));
    ptr += sizeof(uint64_t);
    return buf;
}

NRAMKey tkey_deserialize(const char *buf, Size len) {
    char *ptr = (char *)buf;
    NRAMKey tkey = palloc0(sizeof(NRAMKeyData));

    if (len != sizeof(Oid) + sizeof(uint64_t)) {
        elog(ERROR, "tkey_deserialize: input buffer invalid length");
    }

    memcpy(&tkey->tableOid, ptr, sizeof(Oid));
    ptr += sizeof(Oid);
    memcpy(&tkey->tid, ptr, sizeof(uint64_t));
    ptr += sizeof(uint64_t);
    return tkey;
}


NRAMKey copy_nram_key(NRAMKey src) {
    NRAMKey dst = palloc(sizeof(NRAMKeyData));
    dst->tableOid = src->tableOid;
    dst->tid = src->tid;
    return dst;
}

NRAMValue copy_nram_value(NRAMValue src) {
    Size total_len = offsetof(NRAMValueData, data);
    char *src_ptr = (char *)src + offsetof(NRAMValueData, data);
    int16 nfields = src->nfields;
    NRAMValue dst;

    // First pass to compute total length
    for (int i = 0; i < nfields; i++) {
        NRAMValueFieldData *field = (NRAMValueFieldData *)src_ptr;
        Size field_size = sizeof(NRAMValueFieldData) + field->len;
        total_len += field_size;
        src_ptr += field_size;
    }

    // Allocate and copy
    dst = (NRAMValue)palloc(total_len);
    dst->xact_id = src->xact_id;
    dst->nfields = src->nfields;
    dst->flags = src->flags;

    memcpy((char *)dst + offsetof(NRAMValueData, data),
           (char *)src + offsetof(NRAMValueData, data),
           total_len - offsetof(NRAMValueData, data));

    return dst;
}
