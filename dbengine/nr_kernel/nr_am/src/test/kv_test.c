#include "nram_access/kv.h"
#include "test/kv_test.h"
#include "access/htup_details.h"
#include "utils/builtins.h"
#include "postgres.h"
#include "funcapi.h"

void run_kv_serialization_test(void);

/*
 * This test:
 * 1. Creates a test tuple for schema (int4, text)
 * 2. Serializes to NRAMValue and NRAMKey
 * 3. Deserializes them back
 * 4. Compares results
 */
void run_kv_serialization_test(void) {
    TupleDesc desc;
    HeapTuple tuple, decoded_tuple;

    Datum values[2];
    Datum decoded_values[2];
    uint64_t key_value;

    NRAMKey key, key_copy;
    NRAMValue encoded_value, value_copy;

    bool decoded_isnull[3];
    bool isnull[3] = {false, false, false};

    char *key_buf, *value_buf;
    Size key_len, value_len;
    ItemPointerData tid;

    // Build a synthetic tuple descriptor: (id int, val text)
    desc = CreateTemplateTupleDesc(3);
    TupleDescInitEntry(desc, (AttrNumber) 1, "id", INT4OID, -1, 0);
    TupleDescInitEntry(desc, (AttrNumber) 2, "val", TEXTOID, -1, 0);
    TupleDescInitEntry(desc, (AttrNumber) 3, "desc", TEXTOID, -1, 0);
    values[0] = Int32GetDatum(42);
    values[1] = CStringGetTextDatum("hello");
    values[2] = CStringGetTextDatum("This is a test record");
    BlessTupleDesc(desc);
    tuple = heap_form_tuple(desc, values, isnull);
    tuple->t_tableOid = 0;

    // Serialize to value
    encoded_value = nram_value_serialize_from_tuple(tuple, desc);
    decoded_tuple = deserialize_nram_value_to_tuple(encoded_value, desc);

    // Extract back and compare
    heap_deform_tuple(decoded_tuple, desc, decoded_values, decoded_isnull);

    if (DatumGetInt32(decoded_values[0]) != 42 ||
        strcmp(TextDatumGetCString(decoded_values[1]), "hello") != 0)
        elog(ERROR, "Value encode/decode failed, (%d, %s) != (42, hello)",
            DatumGetInt32(decoded_values[0]), TextDatumGetCString(decoded_values[1]));

    // Serialize key with id as key
    nram_generate_tid(&tid);
    key = nram_key_from_tid(tuple->t_tableOid, &tid);
    key_value = nram_decode_tid(&tid);

    if (key_value != key->tid)
        elog(ERROR, "Key encode/decode failed, %lu != %lu", key_value, key->tid);

    key_buf = tkey_serialize(key, &key_len);
    key_copy = tkey_deserialize(key_buf, key_len);
    if (key_copy->tableOid != 0 || key_copy->tid != key_value)
        elog(ERROR, "tkey_serialize/deserialized failed!");

    pfree(key_buf);
    pfree(key_copy);


    value_buf = tvalue_serialize(encoded_value, &value_len);
    value_copy = tvalue_deserialize(value_buf, value_len);

    if (value_copy->nfields != encoded_value->nfields ||
        value_copy->xact_id != encoded_value->xact_id ||
        memcmp(value_copy->data, encoded_value->data, value_len - offsetof(NRAMValueData, data)) != 0)
            elog(ERROR, "tvalue_serialize/deserialized failed!\nExp: %d,%u,%s\nGot: %d,%u,%s",
                encoded_value->nfields, encoded_value->xact_id, stringify_buff(encoded_value->data, value_len - offsetof(NRAMValueData, data)),
                value_copy->nfields, value_copy->xact_id, stringify_buff(value_copy->data, value_len - offsetof(NRAMValueData, data)));

    pfree(value_buf);
    pfree(value_copy);

    elog(INFO, "Serialization test pass!");
}



/*
 * This test:
 * 1. Creates a test tuple for schema (int4, text)
 * 2. Serializes to NRAMValue and NRAMKey, and make a copy of them.
 * 3. Deserializes them back
 * 4. Compares results
 */
void run_kv_copy_test(void) {
    TupleDesc desc;
    HeapTuple tuple, decoded_tuple;

    Datum values[2];
    Datum decoded_values[2];
    uint64_t key_value;

    NRAMKey key, key_copy;
    NRAMValue encoded_value, value_copy;

    bool decoded_isnull[3];
    bool isnull[3] = {false, false, false};
    ItemPointerData tid;

    // Build a synthetic tuple descriptor: (id int, val text)
    desc = CreateTemplateTupleDesc(3);
    TupleDescInitEntry(desc, (AttrNumber) 1, "id", INT4OID, -1, 0);
    TupleDescInitEntry(desc, (AttrNumber) 2, "val", TEXTOID, -1, 0);
    TupleDescInitEntry(desc, (AttrNumber) 3, "desc", TEXTOID, -1, 0);
    values[0] = Int32GetDatum(42);
    values[1] = CStringGetTextDatum("hello");
    values[2] = CStringGetTextDatum("This is a test record");
    BlessTupleDesc(desc);
    tuple = heap_form_tuple(desc, values, isnull);
    tuple->t_tableOid = 0;

    // Serialize to value
    encoded_value = nram_value_serialize_from_tuple(tuple, desc);
    value_copy = copy_nram_value(encoded_value);
    pfree(encoded_value);
    decoded_tuple = deserialize_nram_value_to_tuple(value_copy, desc);
    heap_deform_tuple(decoded_tuple, desc, decoded_values, decoded_isnull);
    if (DatumGetInt32(decoded_values[0]) != 42 ||
        strcmp(TextDatumGetCString(decoded_values[1]), "hello") != 0)
        elog(ERROR, "Value encode/decode failed, (%d, %s) != (42, hello)",
            DatumGetInt32(decoded_values[0]), TextDatumGetCString(decoded_values[1]));
    pfree(value_copy);
    heap_freetuple(decoded_tuple);

    // Serialize key with id as key
    nram_generate_tid(&tid);
    key = nram_key_from_tid(tuple->t_tableOid, &tid);
    key_copy = copy_nram_key(key);
    pfree(key);
    key_value = nram_decode_tid(&tid);

    if (key_value != key_copy->tid)
        elog(ERROR, "Key encode/decode failed, %lu != %lu", key_value, key_copy->tid);

    elog(INFO, "Copy test pass!");
    pfree(key_copy);
}
