#include "channel_test.h"
#include "ipc/msg.h"
#include "nram_utils/config.h"
#include <sys/time.h>
#include <sys/wait.h>
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"

/*
 * This test:
 * 1. Initializes a KVChannel in shared memory
 * 2. Pushes and Pops multiple messages of varying sizes
 * 3. Tests wrap-around by filling buffer near its end
 * 4. Validates DefaultWriteEntity/DefaultReadEntity with large data
 * 5. Checks blocking Pop behavior by emptying buffer
 */
void run_channel_basic_test(void) {
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
    if (!KVChannelPush(channel, msg1, strlen(msg1) + 1, 0) ||
        !KVChannelPush(channel, msg2, strlen(msg2) + 1, 0) ||
        !KVChannelPush(channel, msg3, strlen(msg3) + 1, 0))
        elog(ERROR, "KVChannelPush failed in initial phase");

    if (!KVChannelPop(channel, buf, strlen(msg1) + 1, 0) ||
        strcmp(buf, msg1) != 0)
        elog(ERROR, "KVChannelPop failed for msg1, got '%s'", buf);

    if (!KVChannelPop(channel, buf, strlen(msg2) + 1, 0) ||
        strcmp(buf, msg2) != 0)
        elog(ERROR, "KVChannelPop failed for msg2, got '%s'", buf);

    if (!KVChannelPop(channel, buf, strlen(msg3) + 1, 0) ||
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
    if (!KVChannelPush(channel, msg1, strlen(msg1) + 1, 0))
        elog(ERROR, "KVChannelPush failed unexpectedly");

    memset(buf, 0, sizeof(buf));
    if (!KVChannelPop(channel, buf, strlen(msg1) + 1, -1))
        elog(ERROR, "Blocking KVChannelPop failed");

    if (strcmp(buf, msg1) != 0)
        elog(ERROR, "Blocking KVChannelPop mismatch: expected '%s', got '%s'", msg1, buf);

    /*
     * Step 4: Buffer full rejection test
     */
    Assert(channel->shared->head == channel->shared->tail);
    channel->shared->head = 0;
    channel->shared->tail = KV_CHANNEL_BUFSIZE - 1;  // Leave only 1 byte space

    if (KVChannelPush(channel, msg2, strlen(msg2) + 1, 0))
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
void run_channel_sequential_test(void) {
    KVChannel* channel;
    char* name = "pseudo_conc_kv_channel";
    char producer_msg[128];
    char consumer_buf[128] = {0};
    int total_messages = 1000000;
    int ncol = 10000;
    int produced = 0;
    int consumed = 0;

    channel = KVChannelInit(name, true);

    /*
     * Interleaved producer/consumer simulation
     */
    while (consumed < total_messages) {
        /* Produce multiple messages */
        for (int i = 0; i < ncol && produced < total_messages; i++) {
            snprintf(producer_msg, sizeof(producer_msg), "msg_%04d", produced % 10000);

            if (!KVChannelPush(channel, producer_msg, strlen(producer_msg) + 1, 0)) {
                NRAM_TEST_INFO("Buffer full!");
                break;  // Buffer full, stop producing
            }
            produced++;
        }

        /* Consume messages if available */
        while (KVChannelPop(channel, consumer_buf, 9, 0)) {
            if (strncmp(consumer_buf, "msg_", 4) != 0)
                elog(ERROR, "Corrupted message during pseudo-concurrency: %s", consumer_buf);

            consumed++;
        }
        if (produced != consumed)
            elog(ERROR, "Produced: %d, Consumed: %d", produced, consumed);
    }

    if (produced != total_messages || consumed != total_messages)
        elog(ERROR, "Mismatch: produced=%d, consumed=%d", produced, consumed);

    KVChannelDestroy(channel);

    elog(INFO, "KV channel sequential test passed!");
}


PGDLLEXPORT void producer_bgw_main(Datum arg) {
    const char *name = "multi_proc_kv_channel";
    int total_messages = 10000;
    double timeout_secs = 60.0;
    char msg[128];
    int produced = 0;
    struct timeval start, now;
    double elapsed;
    KVChannel* channel = KVChannelInit(name, false);
    BackgroundWorkerUnblockSignals();
    NRAM_INFO();

    gettimeofday(&start, NULL);

    while (produced < total_messages) {
        snprintf(msg, sizeof(msg), "msg_%08d-%05d", MyProcPid % 100000000, produced % 100000);
        if (KVChannelPush(channel, msg, strlen(msg) + 1, -1)) {
            produced++;
        } else {
            pg_usleep(1000);  // Sleep 1ms to avoid tight loop
        }

        gettimeofday(&now, NULL);
        elapsed = (now.tv_sec - start.tv_sec) + (now.tv_usec - start.tv_usec) / 1e6;
        if (elapsed > timeout_secs) {
            elog(ERROR, "[Producer %d] Timeout after %.2f seconds", MyProcPid, timeout_secs);
            PrintChannelContent(channel);
        }
    }

    elog(INFO, "[Producer %d] Finished producing %d messages", MyProcPid, produced);
    proc_exit(0);
}


PGDLLEXPORT void consumer_bgw_main(Datum arg) {
    const char *name = "multi_proc_kv_channel";
    int total_messages = 10000;
    double timeout_secs = 60.0;
    char buf[128];
    int consumed = 0;
    struct timeval start, now;
    double elapsed;

    KVChannel* channel = KVChannelInit(name, false);
    BackgroundWorkerUnblockSignals();
    gettimeofday(&start, NULL);
    NRAM_INFO();

    while (consumed < total_messages) {
        if (KVChannelPop(channel, buf, 19, -1)) {
            if (strncmp(buf, "msg_", 4) != 0) {
                elog(ERROR, "[Consumer %d] Corrupted message: %s", MyProcPid, buf);
                PrintChannelContent(channel);
            }
            consumed++;
        } else {
            pg_usleep(1000);  // Sleep 1ms to reduce CPU usage
        }

        gettimeofday(&now, NULL);
        elapsed = (now.tv_sec - start.tv_sec) + (now.tv_usec - start.tv_usec) / 1e6;
        if (elapsed > timeout_secs) {
            elog(ERROR, "[Consumer %d] Timeout after %.2f seconds", MyProcPid, timeout_secs);
            PrintChannelContent(channel);
        }
    }

    elog(INFO, "[Consumer %d] Finished consuming %d messages", MyProcPid, consumed);
    proc_exit(0);
}


/*
 * This test:
 * 1. Initializes KVChannel
 * 2. Spawns producer and consumer threads
 * 3. Performs real parallel message exchange
 * 4. Validates correctness under stress
 */
void run_channel_multiprocess_test(void) {
    KVChannel* channel;
    char* name = "multi_proc_kv_channel";
    pid_t producer_pid, consumer_pid;
    channel = KVChannelInit(name, true);
    BackgroundWorker producer, consumer;
    BackgroundWorkerHandle *handle_producer, *handle_consumer;

    memset(&producer, 0, sizeof(producer));
    producer.bgw_flags = BGWORKER_SHMEM_ACCESS;
    producer.bgw_start_time = BgWorkerStart_ConsistentState;
    producer.bgw_restart_time = BGW_NEVER_RESTART;
    producer.bgw_notify_pid = MyProcPid;
    snprintf(producer.bgw_library_name, BGW_MAXLEN, "nram");
    snprintf(producer.bgw_function_name, BGW_MAXLEN, "producer_bgw_main");
    snprintf(producer.bgw_name, BGW_MAXLEN, "test producer");
    if (!RegisterDynamicBackgroundWorker(&producer, &handle_producer))
        elog(ERROR, "could not register producer");

    memset(&consumer, 0, sizeof(consumer));
    consumer.bgw_flags = BGWORKER_SHMEM_ACCESS;
    consumer.bgw_start_time = BgWorkerStart_ConsistentState;
    consumer.bgw_restart_time = BGW_NEVER_RESTART;
    consumer.bgw_notify_pid = MyProcPid;
    snprintf(consumer.bgw_library_name, BGW_MAXLEN, "nram");
    snprintf(consumer.bgw_function_name, BGW_MAXLEN, "consumer_bgw_main");
    snprintf(consumer.bgw_name, BGW_MAXLEN, "test consumer");
    if (!RegisterDynamicBackgroundWorker(&consumer, &handle_consumer))
        elog(ERROR, "could not register consumer");

    if (WaitForBackgroundWorkerStartup(handle_producer, &producer_pid) != BGWH_STARTED)
        elog(ERROR, "producer did not start");
    if (WaitForBackgroundWorkerStartup(handle_consumer, &consumer_pid) != BGWH_STARTED)
        elog(ERROR, "consumer did not start");


    /* Parent waits */
    if (WaitForBackgroundWorkerShutdown(handle_producer) != BGWH_STOPPED)
        elog(ERROR, "producer did not stop cleanly");

    if (WaitForBackgroundWorkerShutdown(handle_consumer) != BGWH_STOPPED)
        elog(ERROR, "consumer did not stop cleanly");

    Assert(channel->shared->head == channel->shared->tail);

    KVChannelDestroy(channel);
    elog(INFO, "KV channel multi-process test passed");
}


void run_channel_msg_basic_test(void) {
    KVMsg* msg = NewMsg(kv_put, 12345, kv_status_ok, 12345), *recv;
    char* data = "Hello, KV World!";
    KVChannel *channel = KVChannelInit("basic", true);

    msg->header.entitySize = strlen(data) + 1;
    msg->entity = data;
    // PrintKVMsg(&msg);

    if (!KVChannelPushMsg(channel, msg, -1)) {
        elog(ERROR, "Message channel push fail");
    }
    // PrintChannelContent(channel);

    recv = KVChannelPopMsg(channel, -1);

    if(!recv) {
        PrintKVMsg(recv);
        elog(ERROR, "Message channel pop fail");
    }


    Assert(recv->header.op == kv_put);
    Assert(recv->header.relId == 12345);
    Assert(recv->header.entitySize == strlen(data) + 1);
    Assert(recv->entity != NULL);
    Assert(strcmp((char*)recv->entity, data) == 0);
    // Assert(false);

    pfree(recv->entity);
    elog(INFO, "KV channel basic message test passed");
}
