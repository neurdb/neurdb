#include "nram_access/kv.h"
#include "test/kv_test.h"
#include "access/htup_details.h"
#include "utils/builtins.h"
#include "postgres.h"
#include "funcapi.h"
#include "postmaster/bgworker.h"
#include "ipc/msg.h"
#include <sys/time.h>
#include <sys/wait.h>
#include "miscadmin.h"

#include "nram_storage/rocks_service.h"
#include "nram_storage/rocks_handler.h"

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

    Datum values[3];
    Datum decoded_values[3];
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

    elog(INFO, "KV serialization test pass!");
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

    Datum values[3];
    Datum decoded_values[3];
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

    pfree(key_copy);

    elog(INFO, "Value copy test pass!");
}


/*
 * This test:
 * 1. Launches RocksDB service in a child process
 * 2. Sends a PUT request with (key = tableOid:tid), value = ("ABCDEFG")
 * 3. Sends a GET request to verify stored value
 * 4. Validates correctness and cleans up
 */
void run_kv_rocks_service_basic_test(void) {
    KVChannel *channel, *resp_chan;
    NRAMKey key;
    NRAMValue value;
    NRAMValueFieldData *field;
    char *serialized_key, *serialized_value;
    Size key_len, val_len, total_len;
    KVMsg *put_msg, *get_msg, *resp_msg;
    NRAMValue val_out;
    bool ok;    

    channel = KVChannelInit(ROCKSDB_CHANNEL, false);
    resp_chan = KVChannelInit("kv_resp_9999", true);

    // 2. Construct PUT message with key + value
    key = palloc0(sizeof(NRAMKeyData));
    key->tableOid = 1234;
    key->tid = 1;

    value = palloc0(sizeof(NRAMValueData) + sizeof(NRAMValueFieldData) + 7);
    value->xact_id = 1;
    value->nfields = 1;

    field = (NRAMValueFieldData *)value->data;
    field->attnum = 1;
    field->type_oid = TEXTOID;
    field->len = 7;
    memcpy((char *)field + sizeof(NRAMValueFieldData), "ABCDEFG", 7);

    serialized_key = tkey_serialize(key, &key_len);
    serialized_value = tvalue_serialize(value, &val_len);
    total_len = key_len + val_len + sizeof(Size);

    put_msg = NewMsg(kv_put, 1234, kv_status_none, 9999);
    put_msg->header.entitySize = total_len;
    put_msg->entity = palloc(total_len);

    memcpy(put_msg->entity, &key_len, sizeof(Size));
    memcpy((char *)put_msg->entity + sizeof(Size), serialized_key, key_len);
    memcpy((char *)put_msg->entity + sizeof(Size) + key_len, serialized_value, val_len);

    ok = KVChannelPushMsg(channel, put_msg, -1);
    Assert(ok);

    resp_msg = KVChannelPopMsg(resp_chan, -1);
    if (resp_msg == NULL || resp_msg->header.status != kv_status_ok) {
        PrintKVMsg(resp_msg);
        PrintChannelContent(channel);
        PrintChannelContent(resp_chan);
        elog(ERROR, "Rocks service PUT failed");
    }

    // 3. Construct GET message and validate value
    get_msg = NewMsg(kv_get, 1234, kv_status_none, 9999);
    get_msg->header.entitySize = key_len;
    get_msg->entity = serialized_key;

    ok = KVChannelPushMsg(channel, get_msg, -1);
    Assert(ok);

    resp_msg = KVChannelPopMsg(resp_chan, -1);

    if (!resp_msg || resp_msg->header.status != kv_status_ok) {
        PrintKVMsg(resp_msg);
        PrintChannelContent(resp_chan);
        PrintChannelContent(channel);
        elog(ERROR, "Rocks service GET failed");
    }

    // PrintKVMsg(resp_msg);
    Assert(resp_msg->header.entitySize == val_len);
    val_out = tvalue_deserialize((char *)resp_msg->entity, resp_msg->header.entitySize);
    Assert(val_out->xact_id == value->xact_id);
    Assert(val_out->nfields == value->nfields);

    if (memcmp(val_out->data, value->data, 7) != 0)
        elog(ERROR, "Value mismatch, expected 'ABCDEFG' got '%s'", val_out->data);

    // 4. Clean up
    pfree(put_msg->entity);
    pfree(put_msg);
    pfree(get_msg);
    pfree(resp_msg->entity);
    pfree(resp_msg);

    pfree(serialized_key);
    pfree(serialized_value);
    pfree(key);
    pfree(value);
    pfree(val_out);
    KVChannelDestroy(resp_chan);

    elog(INFO, "Rocks service basic test passed!");
}


/*
 * This test:
 * 1. Launches RocksDB service in a child process
 * 2. Sends a PUT request with (key = tableOid:tid), value = ("ABCDEFG")
 * 3. Sends a GET request to verify stored value
 * 4. Validates correctness and cleans up
 */
void run_kv_rocks_client_get_put_test(void) {
    NRAMKey key;
    NRAMValue value, val_out;
    NRAMValueFieldData *field;
    bool ok;

    // 2. Construct key and value
    key = palloc0(sizeof(NRAMKeyData));
    key->tableOid = 1234;
    key->tid = 1;

    value = palloc0(sizeof(NRAMValueData) + sizeof(NRAMValueFieldData) + 7);
    value->xact_id = 1;
    value->nfields = 1;

    field = (NRAMValueFieldData *)value->data;
    field->attnum = 1;
    field->type_oid = TEXTOID;
    field->len = 7;
    memcpy((char *)field + sizeof(NRAMValueFieldData), "ABCDEFG", 7);

    // 3. PUT
    ok = RocksClientPut(key, value);
    Assert(ok);

    // 4. GET
    val_out = RocksClientGet(key);
    Assert(val_out->xact_id == value->xact_id);
    Assert(val_out->nfields == value->nfields);

    if (memcmp(val_out->data, value->data, 7) != 0)
        elog(ERROR, "Value mismatch, expected 'ABCDEFG' got '%s'", val_out->data);

    // 5. Cleanup
    pfree(key);
    pfree(value);
    pfree(val_out);
    CloseRespChannel();

    elog(INFO, "Rocks service client GET PUT test passed!");
}


/*
 * This test:
 * 1. Launches RocksDB service in a child process
 * 2. Sends multiple PUT requests with keys in [tableOid:1] to [tableOid:5]
 * 3. Issues a RANGE_SCAN for [tableOid:2, tableOid:5)
 * 4. Validates returned keys and values are correct and ordered
 * 5. Cleans up resources
 */
void run_kv_rocks_client_range_scan_test(void) {
    NRAMKey key, start_key, end_key, *result_keys;
    NRAMValue value, val_out, *results;
    NRAMValueFieldData *field;
    bool ok;
    uint32_t result_count;
    int table_oid = 5678;

    // 1. Insert 5 keys
    for (int i = 1; i <= 5; i++) {
        key = palloc0(sizeof(NRAMKeyData));
        key->tableOid = table_oid;
        key->tid = i;

        value = palloc0(sizeof(NRAMValueData) + sizeof(NRAMValueFieldData) + 8);
        value->xact_id = i;
        value->nfields = 1;

        field = (NRAMValueFieldData *)value->data;
        field->attnum = 1;
        field->type_oid = TEXTOID;
        field->len = 8;

        char val_str[9];
        snprintf(val_str, sizeof(val_str), "Value%03d", i);
        memcpy((char *)field + sizeof(NRAMValueFieldData), val_str, 8);

        ok = RocksClientPut(key, value);
        Assert(ok);

        pfree(key);
        pfree(value);
    }

    // 2. Setup range: [tableOid:2, tableOid:5)
    start_key = palloc0(sizeof(NRAMKeyData));
    start_key->tableOid = table_oid;
    start_key->tid = 2;

    end_key = palloc0(sizeof(NRAMKeyData));
    end_key->tableOid = table_oid;
    end_key->tid = 5;

    // 3. Perform range scan with keys
    ok = RocksClientRangeScan(start_key, end_key, &result_keys, &results, &result_count);
    Assert(ok);
    Assert(result_count == 3);  // Expect tid = 2, 3, 4

    // 4. Validate keys and results
    for (int i = 0; i < result_count; i++) {
        NRAMKey out_key = result_keys[i];
        val_out = results[i];

        Assert(out_key->tableOid == table_oid);
        Assert(out_key->tid == i + 2);  // tid = 2, 3, 4

        Assert(val_out->xact_id == i + 2);
        Assert(val_out->nfields == 1);

        field = (NRAMValueFieldData *)val_out->data;
        Assert(field->len == 8);

        char expected[9];
        snprintf(expected, sizeof(expected), "Value%03d", i + 2);

        if (memcmp((char *)field + sizeof(NRAMValueFieldData), expected, 8) != 0)
            elog(ERROR, "Range Scan Value mismatch at index %d, expected '%s'", i, expected);
    }

    // 5. Cleanup
    for (int i = 0; i < result_count; i++) {
        pfree(result_keys[i]);
        pfree(results[i]);
    }
    if (result_keys) pfree(result_keys);
    if (results) pfree(results);

    pfree(start_key);
    pfree(end_key);
    CloseRespChannel();

    elog(INFO, "Rocks service client RANGE SCAN test passed!");
}
