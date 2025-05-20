#include "kv.h"
#include "postgres.h"
#include "utils/datum.h"

void nram_key_deserialize(NRAMKey tkey, TupleDesc desc, int *key_attrs,
                          Datum *values) {
    char *pos = tkey->data;

    for (int i = 0; i < tkey->nkeys; i++) {
        int attnum = key_attrs[i];
        Form_pg_attribute attr = TupleDescAttr(desc, attnum);
        Size len = datumGetSize(PointerGetDatum(pos), attr->attbyval, attr->attlen);
        values[i] = PointerGetDatum(palloc(len));
        memcpy(DatumGetPointer(values[i]), pos, len);

        pos += len;
    }
}

NRAMKey nram_key_serialize_from_tuple(HeapTuple tuple, TupleDesc tupdesc,
                                      int *key_attrs, int nkeys) {
    Datum *values = palloc(sizeof(Datum) * nkeys);
    bool *isnull = palloc(sizeof(bool) * nkeys);
    Size *lens = palloc(sizeof(Size) * nkeys);
    Size total_size = sizeof(NRAMKeyData);
    NRAMKey tkey;
    char *pos;

    for (int i = 0; i < nkeys; i++) {
        int attnum = key_attrs[i];
        bool isnull_i;
        Datum d = heap_getattr(tuple, attnum + 1, tupdesc, &isnull_i);
        Form_pg_attribute attr = TupleDescAttr(tupdesc, attnum);

        values[i] = d;
        isnull[i] = isnull_i;
        if (isnull_i) elog(ERROR, "Primary key attribute %d is NULL", attnum);

        lens[i] = datumGetSize(d, attr->attbyval, attr->attlen);
        total_size += lens[i];
    }

    tkey = palloc0(total_size);
    tkey->nkeys = nkeys;

    pos = tkey->data;
    for (int i = 0; i < nkeys; i++) {
        memcpy(pos, DatumGetPointer(values[i]), lens[i]);
        pos += lens[i];
    }

    return tkey;
}

NRAMValue serialize_nram_tuple_to_value(HeapTuple tuple, TupleDesc tupdesc) {
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

    val = palloc0(total_size);
    val->nfields = tupdesc->natts;

    pos = val->data;
    for (int i = 0; i < tupdesc->natts; i++) {
        NRAMValueFieldData *field = (NRAMValueFieldData *)pos;
        field->attnum = i;
        field->type_oid = TupleDescAttr(tupdesc, i)->atttypid;
        field->len = field_lens[i];

        if (field_lens[i] > 0)
            memcpy(field->data, DatumGetPointer(values[i]), field_lens[i]);

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
        int attidx = field->attnum - 1;
        Form_pg_attribute attr = TupleDescAttr(tupdesc, attidx);
        isnull[i] = (field->len == 0);

        if (!isnull[attidx]) {
            if (attr->attbyval) {
                Assert(field->len == attr->attlen);
                memcpy(&values[attidx], field->data, field->len);
            } else {
                char *dataptr = palloc(field->len);
                memcpy(dataptr, field->data, field->len);
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
        NRAMValueFieldData *f = (NRAMValueFieldData *) ptr;
        Size field_size = sizeof(NRAMValueFieldData) + f->len;
        total_len += field_size;
        ptr += field_size;
    }

    buf = malloc(total_len);
    memcpy(buf, &tvalue->nfields, sizeof(int16));

    ptr = tvalue->data;
    write_ptr = buf + sizeof(int16);
    for (int i = 0; i < tvalue->nfields; i++) {
        NRAMValueFieldData *f = (NRAMValueFieldData *) ptr;
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
    NRAMValue tvalue = (NRAMValue) malloc(len);

    memcpy(&nfields, buf, sizeof(int16));
    memcpy(tvalue, buf, len);  // includes both nfields and all field data
    return tvalue;
}


char *tkey_serialize(NRAMKey tkey, Size *out_len) {
    *out_len = sizeof(int16) + sizeof(Size) + tkey->length;
    char *buf = malloc(*out_len);
    char *ptr = buf;

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
    NRAMKey tkey;
    
    if (len < sizeof(int16) + sizeof(Size)) {
        elog(ERROR, "tkey_deserialize: input buffer too short");
    }

    memcpy(&nkeys, ptr, sizeof(int16));
    ptr += sizeof(int16);

    memcpy(&datalen, ptr, sizeof(Size));
    ptr += sizeof(Size);

    if (len < sizeof(int16) + sizeof(Size) + datalen) {
        elog(ERROR, "tkey_deserialize: inconsistent data length");
    }

    tkey = (NRAMKey) malloc(sizeof(NRAMKeyData) + datalen);
    tkey->nkeys = nkeys;
    tkey->length = datalen;
    memcpy(tkey->data, ptr, datalen);
    return tkey;
}
