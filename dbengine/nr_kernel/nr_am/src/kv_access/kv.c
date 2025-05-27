#include "kv.h"
#include "utils/datum.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "funcapi.h"


char *stringify_nram_key(NRAMKey key, TupleDesc desc, int *key_attrs) {
    char *pos = (char *)key + sizeof(NRAMKeyData);
    char *end = pos + key->length;
    StringInfoData buf;
    bool *is_null = NULL;

    initStringInfo(&buf);
    appendStringInfo(&buf,
                     "[NRAMKey] nkeys = %d, length = %zu, tableOid = %u, ",
                     key->nkeys, key->length, key->tableOid);
    is_null = palloc(key->nkeys * sizeof(bool));

    for (int i = 0; i < key->nkeys; i++) {
        int attnum = key_attrs[i] - 1;
        Form_pg_attribute attr = TupleDescAttr(desc, attnum);
        Datum val = datumRestore(&pos, &is_null[i]);
        Oid typoutput;
        char *value_str;
        bool typIsVarlena;


        if (is_null[i]) {
            appendStringInfo(&buf, "  {Key[%d] (attnum=%d, type=%u): NIL} ", i,
                            attnum + 1, attr->atttypid);
        } else {
            getTypeOutputInfo(attr->atttypid, &typoutput, &typIsVarlena);
            value_str = OidOutputFunctionCall(typoutput, val);
            appendStringInfo(&buf, "  {Key[%d] (attnum=%d, type=%u): %s} ", i,
                            attnum + 1, attr->atttypid, value_str);
        }
    }

    pfree(is_null);
    if (pos != end)
        elog(ERROR,
                "Buffer overflow: miss alignment the offset is %d\nThe current string is %s",
                (int)(pos-end), buf.data);

    return buf.data;
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

void nram_key_deserialize(NRAMKey tkey, TupleDesc desc, int *key_attrs,
                          Datum *values, bool *is_null) {
    char *pos = (char *)tkey + sizeof(NRAMKeyData);

    for (int i = 0; i < tkey->nkeys; i++) values[i] = datumRestore(&pos, &is_null[i]);

    if (pos - (char *)tkey - sizeof(NRAMKeyData) != tkey->length)
        elog(ERROR, "[nram_key_deserialize]: miss alignment the offset is expected: %zu, real: %d",
                tkey->length,
                (int)(pos - (char *)tkey - sizeof(NRAMKeyData)));
}

NRAMKey nram_key_serialize_from_tuple(HeapTuple tuple, TupleDesc tupdesc,
                                      int *key_attrs, int nkeys) {
    Datum *values = palloc(sizeof(Datum) * nkeys);
    bool *isnull = palloc(sizeof(bool) * nkeys);
    Size *lens = palloc(sizeof(Size) * nkeys);
    Size total_size = sizeof(NRAMKeyData);
    NRAMKey tkey;
    char *pos;

    // Collect key values and compute lengths
    for (int i = 0; i < nkeys; i++) {
        int attnum = key_attrs[i] - 1;  // key_attrs is 1-based
        bool isnull_i;
        Datum d = heap_getattr(tuple, attnum + 1, tupdesc, &isnull_i);
        Form_pg_attribute attr = TupleDescAttr(tupdesc, attnum);

        if (isnull_i)
            elog(ERROR, "Primary key attribute %d is NULL", key_attrs[i]);

        values[i] = d;
        isnull[i] = isnull_i;
        lens[i] = datumEstimateSpace(values[i], isnull[i], attr->attbyval, attr->attlen);
        total_size += lens[i];
        // NRAM_TEST_INFO("Counting attr %d, len=%zu", key_attrs[i], lens[i]);
    }

    // Allocate and fill the key structure
    tkey = palloc(total_size);
    tkey->tableOid = tuple->t_tableOid;
    tkey->nkeys = nkeys;
    tkey->length = total_size - sizeof(NRAMKeyData);

    pos = (char *)tkey + sizeof(NRAMKeyData);
    for (int i = 0; i < nkeys; i++) {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, key_attrs[i] - 1);
        datumSerialize(values[i], isnull[i], attr->attbyval, attr->attlen, &pos);
        NRAM_TEST_INFO("Serializing attr %d, len=%zu, offset=%ld",
                       key_attrs[i], lens[i], pos - (char *)tkey - sizeof(NRAMKeyData));
    }

    if (pos - (char*)tkey != total_size)
        elog(ERROR, "[nram_key_serialize_from_tuple]: wrong key serialization length mismatch: expected %zu: real %d",
            total_size,
            (int)(pos - (char*)tkey));

    return tkey;
}

NRAMValue nram_value_serialize_from_tuple(HeapTuple tuple, TupleDesc tupdesc) {
    Datum *values = palloc(sizeof(Datum) * tupdesc->natts);
    bool *isnull = palloc(sizeof(bool) * tupdesc->natts);
    NRAMValue val;
    Size total_size, *lens;
    char *pos;

    heap_deform_tuple(tuple, tupdesc, values, isnull);

    // Estimate space needed
    total_size = sizeof(NRAMValueData);
    lens = palloc(sizeof(Size) * tupdesc->natts);

    for (int i = 0; i < tupdesc->natts; i++) {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, i);
        lens[i] = datumEstimateSpace(values[i], isnull[i], attr->attbyval, attr->attlen);
        total_size += lens[i] + sizeof(NRAMValueFieldData);
    }

    val = (NRAMValueData *)palloc0(total_size);
    val->nfields = tupdesc->natts;

    pos = (char*)val + sizeof(NRAMValueData);
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

    char *pos = (char*)val + sizeof(NRAMValueData);
    for (int i = 0; i < val->nfields; i++) {
        pos += sizeof(NRAMValueFieldData);
        values[i] = datumRestore(&pos, &is_null[i]);
    }

    return heap_form_tuple(tupdesc, values, is_null);
}

char *tvalue_serialize(NRAMValue tvalue, Size *out_len) {
    char *ptr = (char*)tvalue + sizeof(NRAMValueData), *buf, *write_ptr;
    Size total_len = sizeof(int16);  // nfields
    for (int i = 0; i < tvalue->nfields; i++) {
        NRAMValueFieldData *f = (NRAMValueFieldData *)ptr;
        Size field_size = sizeof(NRAMValueFieldData) + f->len;
        total_len += field_size;
        ptr += field_size;
    }

    buf = palloc(total_len);
    memcpy(buf, &tvalue->nfields, sizeof(int16));

    ptr = (char*)tvalue + sizeof(NRAMValueData);
    write_ptr = buf + sizeof(int16);
    for (int i = 0; i < tvalue->nfields; i++) {
        NRAMValueFieldData *f = (NRAMValueFieldData *)ptr;
        Size field_size = sizeof(NRAMValueFieldData) + f->len;
        memcpy(write_ptr, f, field_size);
        ptr += field_size;
        write_ptr += field_size;
    }

    *out_len = total_len;
    return buf;
}

NRAMValue tvalue_deserialize(char *buf, Size len) {
    int16 nfields;
    NRAMValue tvalue = (NRAMValue)palloc(len);

    memcpy(&nfields, buf, sizeof(int16));
    memcpy(tvalue, buf, len);  // includes both nfields and all field data
    return tvalue;
}

// Layout [Oid tableOid][int16 nkeys][Size length][data...]
char *tkey_serialize(NRAMKey tkey, Size *out_len) {
    char *buf, *ptr;

    *out_len = sizeof(Oid) + sizeof(int16) + sizeof(Size) + tkey->length;
    buf = palloc0(*out_len);
    ptr = buf;

    memcpy(ptr, &tkey->tableOid, sizeof(Oid));
    ptr += sizeof(Oid);

    memcpy(ptr, &tkey->nkeys, sizeof(int16));
    ptr += sizeof(int16);

    memcpy(ptr, &tkey->length, sizeof(Size));
    ptr += sizeof(Size);

    memcpy(ptr, (char*)tkey + sizeof(NRAMKeyData), tkey->length);
    return buf;
}

NRAMKey tkey_deserialize(char *buf, Size len) {
    char *ptr = buf;
    int16 nkeys;
    Size datalen;
    Oid tableOid;
    NRAMKey tkey;

    if (len < sizeof(Oid) + sizeof(int16) + sizeof(Size)) {
        elog(ERROR, "tkey_deserialize: input buffer too short");
    }

    memcpy(&tableOid, ptr, sizeof(Oid));
    ptr += sizeof(Oid);

    memcpy(&nkeys, ptr, sizeof(int16));
    ptr += sizeof(int16);

    memcpy(&datalen, ptr, sizeof(Size));
    ptr += sizeof(Size);

    if (len < sizeof(Oid) + sizeof(int16) + sizeof(Size) + datalen) {
        elog(ERROR, "tkey_deserialize: inconsistent data length: expected %zu, got %zu",
            len, sizeof(Oid) + sizeof(int16) + sizeof(Size) + datalen);
    }

    tkey = (NRAMKey)palloc(sizeof(NRAMKeyData) + datalen);
    tkey->tableOid = tableOid;
    tkey->nkeys = nkeys;
    tkey->length = datalen;
    memcpy((char*)tkey + sizeof(NRAMKeyData), ptr, datalen);
    return tkey;
}
