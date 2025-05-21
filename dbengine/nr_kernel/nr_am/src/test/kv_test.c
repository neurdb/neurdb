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
    
    NRAMKey key;
    NRAMValue encoded_value;
    int key_attrs[] = {1};

    bool decoded_isnull[2];
    bool isnull[2] = {false, false};


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
    NRAM_TEST_INFO("Serialized.");

    nram_key_deserialize(key, desc, key_attrs, &deserialized_key_val);

    if (DatumGetInt32(deserialized_key_val) != 42)
        elog(ERROR, "Key encode/decode failed, %d != 42", DatumGetInt32(deserialized_key_val));
    else
        NRAM_TEST_INFO("NRAMKey encode/decode works correctly!");
    
    NRAM_TEST_INFO("-------- End --------");
}
