#include "postgres.h"
#include "kv_access/kv.h"
#include "test/kv_test.h"

void run_kv_serialization_test();

/*
 * This test:
 * 1. Creates a test tuple for schema (int4, text)
 * 2. Serializes to NRAMValue and NRAMKey
 * 3. Deserializes them back
 * 4. Compares results
 */
void run_kv_serialization_test() {
    TupleDesc desc;
    HeapTuple tuple, decoded_tuple;
    
    Datum values[2];
    Datum decoded_values[2];
    Datum deserialized_key_val;
    
    NRAMKey key;
    NRAMValue encoded_value;
    int key_attrs[] = {0};

    bool decoded_isnull[2];
    bool isnull[2] = {false, false};


    elog(INFO, "Running nram tuple serialization/deserialization test...");

    // Build a synthetic tuple descriptor: (id int, val text)
    desc = CreateTemplateTupleDesc(2);
    TupleDescInitEntry(desc, (AttrNumber) 1, "id", INT4OID, -1, 0);
    TupleDescInitEntry(desc, (AttrNumber) 2, "val", TEXTOID, -1, 0);

    // Create test tuple: (id=42, val="hello")
    values[0] = Int32GetDatum(42);
    values[1] = CStringGetDatum("hello");
    tuple = heap_form_tuple(desc, values, isnull);

    // Serialize to value
    encoded_value = serialize_nram_tuple_to_value(tuple, desc);
    decoded_tuple = deserialize_nram_value_to_tuple(encoded_value, desc);

    // Extract back and compare
    heap_deform_tuple(decoded_tuple, desc, decoded_values, decoded_isnull);

    if (DatumGetInt32(decoded_values[0]) != 42 ||
        strcmp(DatumGetCString(decoded_values[1]), "hello") != 0)
        elog(ERROR, "Value encode/decode failed");

    // Serialize key with id as key
    key = nram_key_serialize_from_tuple(tuple, desc, key_attrs, 1);

    nram_key_deserialize(key, desc, key_attrs, &deserialized_key_val);

    if (DatumGetInt32(deserialized_key_val) != 42)
        elog(ERROR, "Key encode/decode failed");

    elog(INFO, "Test passed: encode/decode for key and value work correctly.");
}
