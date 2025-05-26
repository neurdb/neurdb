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

    initStringInfo(&buf);
    appendStringInfo(&buf,
                     "[NRAMKey] nkeys = %d, length = %zu, tableOid = %u, ",
                     key->nkeys, key->length, key->tableOid);

    for (int i = 0; i < key->nkeys; i++) {
        int attnum = key_attrs[i] - 1;
        Form_pg_attribute attr = TupleDescAttr(desc, attnum);
        Datum val;
        Oid typoutput;
        char *value_str;
        bool typIsVarlena;

        if (pos + attr->attlen > end) {
            elog(ERROR,
                 "Buffer overflow: field %d at pos=%p with attlen=%d exceeds "
                 "key length %zu",
                 i, pos, attr->attlen, key->length);
        }

        if (attr->attbyval) {
            memcpy(&val, pos, attr->attlen);
            pos += attr->attlen;
        } else {
            Size len = datumGetSize(PointerGetDatum(pos), attr->attbyval,
                                    attr->attlen);
            char *copy = palloc(len);
            memcpy(copy, pos, len);
            val = PointerGetDatum(copy);
            pos += len;
        }

        getTypeOutputInfo(attr->atttypid, &typoutput, &typIsVarlena);

        value_str = OidOutputFunctionCall(typoutput, val);
        appendStringInfo(&buf, "  {Key[%d] (attnum=%d, type=%u): %s} ", i,
                         attnum + 1, attr->atttypid, value_str);
    }

    if (pos != end)
        elog(ERROR,
                "Buffer overflow: miss alignment the offset is %zu, current string is %s",
                pos-end, buf.data);


    return buf.data;  // returned string is palloc'd
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
                          Datum *values) {
    char *pos = (char *)tkey + sizeof(NRAMKeyData);
    char *end = pos + tkey->length;

    for (int i = 0; i < tkey->nkeys; i++) {
        int attnum = key_attrs[i] - 1;
        Form_pg_attribute attr = TupleDescAttr(desc, attnum);
        if (attr->attbyval) {
            Datum val = 0;
            memcpy(&val, pos, attr->attlen);
            values[i] = val;
        } else {
            Size len = datumGetSize(PointerGetDatum(pos), attr->attbyval,
                                    attr->attlen);
            char *copy = palloc(len);
            memcpy(copy, pos, len);
            values[i] = PointerGetDatum(copy);
        }
        pos += attr->attbyval ? attr->attlen
                              : datumGetSize(PointerGetDatum(pos),
                                             attr->attbyval, attr->attlen);
    }

    if (pos != end)
        elog(ERROR, "Deserialization: miss alignment the offset is %zu",
                pos-end);
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
        lens[i] = datumGetSize(d, attr->attbyval, attr->attlen);
        total_size += lens[i];

        // NRAM_TEST_INFO("Counting attr %d, len=%zu", key_attrs[i], lens[i]);
    }

    // Allocate and fill the key structure
    tkey = palloc0(total_size);
    tkey->tableOid = tuple->t_tableOid;
    tkey->nkeys = nkeys;
    tkey->length = total_size - sizeof(NRAMKeyData);

    pos = (char *)tkey + sizeof(NRAMKeyData);
    for (int i = 0; i < nkeys; i++) {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, key_attrs[i] - 1);
        // NRAM_TEST_INFO("Serializing attr %d, len=%zu, offset=%ld",
        //                key_attrs[i], lens[i], pos - (char *)tkey->data);

        if (attr->attbyval) {
            memcpy(pos, &values[i], attr->attlen);  // safe for int, float, etc.
        } else {
            void *src = DatumGetPointer(values[i]);
            if (!src)
                elog(ERROR,
                     "NULL by-ref pointer in key serialization for attr %d",
                     key_attrs[i]);
            memcpy(pos, src, lens[i]);
        }

        pos += lens[i];
    }

    return tkey;
}

NRAMValue nram_value_serialize_from_tuple(HeapTuple tuple, TupleDesc tupdesc) {
    Datum *values = palloc(sizeof(Datum) * tupdesc->natts);
    bool *isnull = palloc(sizeof(bool) * tupdesc->natts);
    NRAMValue val;
    Size total_size, *field_lens;
    char *pos;

    heap_deform_tuple(tuple, tupdesc, values, isnull);

    // Estimate space needed
    total_size = sizeof(NRAMValueData);
    field_lens = palloc(sizeof(Size) * tupdesc->natts);

    for (int i = 0; i < tupdesc->natts; i++) {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, i);
        if (isnull[i])
            field_lens[i] = 0;
        else
            field_lens[i] =
                datumGetSize(values[i], attr->attbyval, attr->attlen);
        total_size += sizeof(NRAMValueFieldData) + field_lens[i];
    }

    val = (NRAMValueData *)palloc0(total_size);
    val->nfields = tupdesc->natts;

    pos = val->data;
    for (int i = 0; i < tupdesc->natts; i++) {
        NRAMValueFieldData *field = (NRAMValueFieldData *)pos;
        Form_pg_attribute attr = TupleDescAttr(tupdesc, i);

        field->attnum = i;
        field->type_oid = TupleDescAttr(tupdesc, i)->atttypid;
        field->len = field_lens[i];

        // NRAM_TEST_INFO("Serializing attr %d, len=%u, offset=%ld",
        // field->attnum, field->len, pos - (char *)val->data);
        if ((pos + sizeof(NRAMValueFieldData) + field_lens[i]) >
            ((char *)val + total_size)) {
            elog(ERROR,
                 "WRITE OUT OF BOUNDS at field %d! Trying to write beyond "
                 "total_size=%zu",
                 i, total_size);
        }

        if (field_lens[i] > 0) {
            char *field_data = (char *)field + sizeof(NRAMValueFieldData);
            if (attr->attbyval) {
                memcpy(field_data, &values[i], sizeof(Datum));
            } else {
                void *src = DatumGetPointer(values[i]);
                if (!src) elog(ERROR, "NULL Datum pointer for attr %d", i);
                memcpy(field_data, src, field_lens[i]);
            }
        }

        pos += sizeof(NRAMValueFieldData) + field_lens[i];
    }

    return val;
}

HeapTuple deserialize_nram_value_to_tuple(NRAMValue val, TupleDesc tupdesc) {
    Datum *values = palloc0(sizeof(Datum) * val->nfields);
    bool *isnull = palloc0(sizeof(bool) * val->nfields);

    char *pos = val->data;
    for (int i = 0; i < val->nfields; i++) {
        NRAMValueFieldData *field = (NRAMValueFieldData *)pos;
        int attidx = field->attnum;
        Form_pg_attribute attr = TupleDescAttr(tupdesc, attidx);

        isnull[attidx] = (field->len == 0);
        // NRAM_TEST_INFO("Deserialized attr %d (attidx %d), len=%u, isnull=%d",
        //      field->attnum, attidx, field->len, isnull[attidx]);

        if (!isnull[attidx]) {
            char *field_data = (char *)field + sizeof(NRAMValueFieldData);

            if (attr->attbyval) {
                if (field->len != attr->attlen)
                    elog(ERROR,
                         "Mismatch byvalue type: field->len=%d vs "
                         "att->attlen=%d (attbyval=%d)",
                         field->len, attr->attlen, attr->attbyval);
                memcpy(&values[attidx], field_data, field->len);
            } else {
                char *dataptr = palloc(field->len);
                memcpy(dataptr, field_data, field->len);
                values[attidx] = PointerGetDatum(dataptr);
            }
        }

        pos += sizeof(NRAMValueFieldData) + field->len;
    }

    return heap_form_tuple(tupdesc, values, isnull);
}

char *tvalue_serialize(NRAMValue tvalue, Size *out_len) {
    char *ptr = tvalue->data, *buf, *write_ptr;
    Size total_len = sizeof(int16);  // nfields
    for (int i = 0; i < tvalue->nfields; i++) {
        NRAMValueFieldData *f = (NRAMValueFieldData *)ptr;
        Size field_size = sizeof(NRAMValueFieldData) + f->len;
        total_len += field_size;
        ptr += field_size;
    }

    buf = palloc(total_len);
    memcpy(buf, &tvalue->nfields, sizeof(int16));

    ptr = tvalue->data;
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

    memcpy(ptr, tkey->data, tkey->length);
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
    memcpy(tkey->data, ptr, datalen);
    return tkey;
}
