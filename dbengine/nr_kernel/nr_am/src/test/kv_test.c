#include "nram_access/kv.h"
#include "test/kv_test.h"
#include "access/htup_details.h"
#include "utils/builtins.h"
#include "postgres.h"
#include "funcapi.h"
#include "ipc/msg.h"
#include <pthread.h>

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
 * 1. Initializes a KVChannel in shared memory
 * 2. Pushes and Pops multiple messages of varying sizes
 * 3. Tests wrap-around by filling buffer near its end
 * 4. Validates DefaultWriteEntity/DefaultReadEntity with large data
 * 5. Checks blocking Pop behavior by emptying buffer
 */
void run_kv_channel_basic_test(void) {
    KVChannel* channel;
    char* name = "complex_kv_channel";
    char msg1[] = "FirstMessage";
    char msg2[] = "SecondMessage";
    char msg3[] = "ThirdMessage";
    char buf[256] = {0};
    uint64 offset = 0;
    char big_msg[KV_CHANNEL_BUFSIZE / 2] = {0};
    char big_out[KV_CHANNEL_BUFSIZE / 2] = {0};

    // Fill big message with identifiable pattern
    memset(big_msg, 'A', sizeof(big_msg) - 1);
    big_msg[sizeof(big_msg) - 1] = '\0';

    channel = KVChannelInit(name, true);

    /*
     * Step 1: Push and Pop multiple small messages
     */
    if (!KVChannelPush(channel, msg1, strlen(msg1) + 1, false) ||
        !KVChannelPush(channel, msg2, strlen(msg2) + 1, false) ||
        !KVChannelPush(channel, msg3, strlen(msg3) + 1, false))
        elog(ERROR, "KVChannelPush failed in initial phase");

    if (!KVChannelPop(channel, buf, strlen(msg1) + 1, false) ||
        strcmp(buf, msg1) != 0)
        elog(ERROR, "KVChannelPop failed for msg1, got '%s'", buf);

    if (!KVChannelPop(channel, buf, strlen(msg2) + 1, false) ||
        strcmp(buf, msg2) != 0)
        elog(ERROR, "KVChannelPop failed for msg2, got '%s'", buf);

    if (!KVChannelPop(channel, buf, strlen(msg3) + 1, false) ||
        strcmp(buf, msg3) != 0)
        elog(ERROR, "KVChannelPop failed for msg3, got '%s'", buf);

    /*
     * Step 2: Wrap-around test with large message
     * Push large message to near end, force wrap
     */
    offset = KV_CHANNEL_BUFSIZE - 5;
    channel->shared->tail = offset;
    channel->shared->head = offset;
    DefaultWriteEntity(channel, &channel->shared->tail, big_msg, sizeof(big_msg));
    DefaultReadEntity(channel, &channel->shared->head, big_out, sizeof(big_msg));

    if (strncmp(big_msg, big_out, sizeof(big_msg)) != 0)
        elog(ERROR, "Wrap-around entity RW mismatch");

    /*
     * Step 3: Blocking Pop scenario
     */
    if (!KVChannelPush(channel, msg1, strlen(msg1) + 1, false))
        elog(ERROR, "KVChannelPush failed unexpectedly");

    memset(buf, 0, sizeof(buf));
    if (!KVChannelPop(channel, buf, strlen(msg1) + 1, true))
        elog(ERROR, "Blocking KVChannelPop failed");

    if (strcmp(buf, msg1) != 0)
        elog(ERROR, "Blocking KVChannelPop mismatch: expected '%s', got '%s'", msg1, buf);

    /*
     * Step 4: Buffer full rejection test
     */
    channel->shared->head = 0;
    channel->shared->tail = KV_CHANNEL_BUFSIZE - 1;  // Leave only 1 byte space

    if (KVChannelPush(channel, msg2, strlen(msg2) + 1, false))
        elog(ERROR, "KVChannelPush should have failed due to no space");

    KVChannelDestroy(channel);

    elog(INFO, "KV channel test passed!");
}


/*
 * This test:
 * 1. Initializes KVChannel
 * 2. Simulates interleaved Push/Pop operations as pseudo-concurrent tasks
 * 3. Forces buffer wrap-around
 * 4. Validates synchronization logic under stress
 */
void run_kv_channel_sequential_test(void) {
    KVChannel* channel;
    char* name = "pseudo_conc_kv_channel";
    char producer_msg[128];
    char consumer_buf[128] = {0};
    int total_messages = 1000;
    int produced = 0;
    int consumed = 0;

    channel = KVChannelInit(name, true);

    /*
     * Interleaved producer/consumer simulation
     */
    while (consumed < total_messages) {
        /* Produce multiple messages */
        for (int i = 0; i < 10 && produced < total_messages; i++) {
            snprintf(producer_msg, sizeof(producer_msg), "msg_%04d", produced);

            if (!KVChannelPush(channel, producer_msg, strlen(producer_msg) + 1, false)) {
                break;  // Buffer full, stop producing
            }
            produced++;
        }

        /* Consume messages if available */
        while (KVChannelPop(channel, consumer_buf, 9, false)) {
            if (strncmp(consumer_buf, "msg_", 4) != 0)
                elog(ERROR, "Corrupted message during pseudo-concurrency: %s", consumer_buf);
            // else
            //     NRAM_TEST_INFO("consumer received message %s", consumer_buf);

            consumed++;
        }

        /* Random sleep to mimic timing gaps */
        if (produced % 100 == 0 || consumed % 100 == 0)
            pg_usleep(1000);
    }

    if (produced != total_messages || consumed != total_messages)
        elog(ERROR, "Mismatch: produced=%d, consumed=%d", produced, consumed);

    KVChannelDestroy(channel);

    elog(INFO, "KV channel sequential test passed!");
}


/*
 * This test:
 * 1. Initializes KVChannel
 * 2. Spawns producer and consumer threads
 * 3. Performs real parallel message exchange
 * 4. Validates correctness under stress
 */
void run_kv_channel_multiprocess_test(void) {
    KVChannel* channel;
    char* name = "multi_proc_kv_channel";
    int total_messages = 1000;
    pid_t producer_pid, consumer_pid;

    channel = KVChannelInit(name, true);

    producer_pid = fork();
    if (producer_pid == 0) {
        /* Producer process */
        channel = KVChannelInit(name, false);
        char msg[128];
        int produced = 0;
        // NRAM_TEST_INFO("Producer process started");

        while (produced < total_messages) {
            snprintf(msg, sizeof(msg), "msg_%04d", produced);
            if (KVChannelPush(channel, msg, strlen(msg) + 1, true))
                produced++;
        }
        // NRAM_TEST_INFO("Producer finished");
        _exit(0);
    }

    consumer_pid = fork();
    if (consumer_pid == 0) {
        /* Consumer process */
        channel = KVChannelInit(name, false);
        char buf[128];
        int consumed = 0;
        // NRAM_TEST_INFO("Consumer process started");

        while (consumed < total_messages) {
            if (KVChannelPop(channel, buf, 9, true)) {
                if (strncmp(buf, "msg_", 4) != 0)
                    elog(ERROR, "Corrupted message: %s", buf);
                consumed++;
            }
        }
        // NRAM_TEST_INFO("Consumer finished");
        _exit(0);
    }

    /* Parent waits */
    int status;
    waitpid(producer_pid, &status, 0);
    Assert(status == 0);
    waitpid(consumer_pid, &status, 0);
    Assert(status == 0);

    KVChannelDestroy(channel);
    elog(INFO, "KV channel multi-process test passed");
}

