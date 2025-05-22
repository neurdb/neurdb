#include "postgres.h"
#include "kv_access/kv.h"
#include "test/kv_test.h"
#include "utils/builtins.h"

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
    Datum deserialized_key_val;
    
    NRAMKey key, key_copy;
    NRAMValue encoded_value, value_copy;
    int key_attrs[] = {1};

    bool decoded_isnull[2];
    bool isnull[2] = {false, false};

    char *key_buf, *value_buf;
    Size key_len, value_len;


    NRAM_TEST_INFO("-------- Start --------");

    // Build a synthetic tuple descriptor: (id int, val text)
    desc = CreateTemplateTupleDesc(2);
    TupleDescInitEntry(desc, (AttrNumber) 1, "id", INT4OID, -1, 0);
    TupleDescInitEntry(desc, (AttrNumber) 2, "val", TEXTOID, -1, 0);
    values[0] = Int32GetDatum(42);
    values[1] = CStringGetTextDatum("hello");
    BlessTupleDesc(desc);
    tuple = heap_form_tuple(desc, values, isnull);

    // Serialize to value
    encoded_value = nram_value_serialize_from_tuple(tuple, desc);
    decoded_tuple = deserialize_nram_value_to_tuple(encoded_value, desc);

    // Extract back and compare
    heap_deform_tuple(decoded_tuple, desc, decoded_values, decoded_isnull);

    if (DatumGetInt32(decoded_values[0]) != 42 ||
        strcmp(TextDatumGetCString(decoded_values[1]), "hello") != 0)
        elog(ERROR, "Value encode/decode failed, (%d, %s) != (42, hello)", 
            DatumGetInt32(decoded_values[0]), TextDatumGetCString(decoded_values[1]));
   else
        NRAM_TEST_INFO("NRAMValue encode/decode works correctly!");

    // Serialize key with id as key
    key = nram_key_serialize_from_tuple(tuple, desc, key_attrs, 1);
    nram_key_deserialize(key, desc, key_attrs, &deserialized_key_val);

    if (DatumGetInt32(deserialized_key_val) != 42)
        elog(ERROR, "Key encode/decode failed, %d != 42", DatumGetInt32(deserialized_key_val));
    else
        NRAM_TEST_INFO("NRAMKey encode/decode works correctly!");

    // Key/value buf tests.
    key_buf = tkey_serialize(key, &key_len);
    key_copy = tkey_deserialize(key_buf, key_len);
    if (key_copy->nkeys != 1 || key_copy->length != key->length ||
        memcmp(key_copy->data, key->data, key->length) != 0)
        elog(ERROR, "tkey_serialize/deserialized failed!");
    else
        NRAM_TEST_INFO("tkey_serialize/deserialized works correctly!");
    
    pfree(key_buf);
    pfree(key_copy);


    value_buf = tvalue_serialize(encoded_value, &value_len);
    value_copy = tvalue_deserialize(value_buf, value_len);

    if (value_copy->nfields != encoded_value->nfields ||
        memcmp(value_copy->data, encoded_value->data,
               value_len - offsetof(NRAMValueData, data)) != 0)
        elog(ERROR, "tvalue_serialize/deserialized failed!");
    else
        NRAM_TEST_INFO("tvalue_serialize/deserialized works correctly!");

    pfree(value_buf);
    pfree(value_copy);

    
    NRAM_TEST_INFO("-------- End --------");
}
