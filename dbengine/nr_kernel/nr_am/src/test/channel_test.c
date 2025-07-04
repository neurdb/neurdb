#include "channel_test.h"
#include "ipc/msg.h"
#include "nram_utils/config.h"
#include <sys/time.h>
#include <sys/wait.h>
#include "miscadmin.h"

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
    Assert(channel->shared->head == channel->shared->tail);
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
void run_channel_sequential_test(void) {
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
                NRAM_TEST_INFO("Buffer full!");
                break;  // Buffer full, stop producing
            }
            produced++;
        }

        /* Consume messages if available */
        while (KVChannelPop(channel, consumer_buf, 9, false)) {
            if (strncmp(consumer_buf, "msg_", 4) != 0)
                elog(ERROR, "Corrupted message during pseudo-concurrency: %s", consumer_buf);

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
void run_channel_multiprocess_test(void) {
    KVChannel* channel;
    char* name = "multi_proc_kv_channel";
    pid_t producer_pid, consumer_pid;
    int total_messages = 10, status = 0;
    double timeout_secs = 2.0;

    channel = KVChannelInit(name, true);

    producer_pid = fork();
    if (producer_pid == 0) {
        /* Producer process */
        char msg[128];
        int produced = 0;
        struct timeval start, now;
        double elapsed;

        channel = KVChannelInit(name, false);

        gettimeofday(&start, NULL);

        while (produced < total_messages) {
            snprintf(msg, sizeof(msg), "msg_%08d-%05d", MyProcPid, produced);
            if (KVChannelPush(channel, msg, strlen(msg) + 1, true)) {
                produced++;
                // elog(INFO, "[Producer %d] Push message %d", MyProcPid, produced);
                // PrintChannelContent(channel);
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
        _exit(0);
    }

    consumer_pid = fork();
    if (consumer_pid == 0) {
        /* Consumer process */
        char buf[128];
        int consumed = 0;
        struct timeval start, now;
        double elapsed;

        channel = KVChannelInit(name, false);
        gettimeofday(&start, NULL);

        while (consumed < total_messages) {
            if (KVChannelPop(channel, buf, 19, true)) {
                if (strncmp(buf, "msg_", 4) != 0) {
                    elog(ERROR, "[Consumer %d] Corrupted message: %s", MyProcPid, buf);
                    PrintChannelContent(channel);
                }
                // else {
                //     elog(INFO, "[Consumer %d] Got message: %s", MyProcPid, buf);
                // }
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
        _exit(0);
    }

    /* Parent waits */
    waitpid(producer_pid, &status, 0);
    Assert(status == 0);
    waitpid(consumer_pid, &status, 0);
    Assert(status == 0);

    // PrintChannelContent(channel);
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

    if (!KVChannelPushMsg(channel, msg, true)) {
        elog(ERROR, "Message channel push fail");
    }
    // PrintChannelContent(channel);

    recv = KVChannelPopMsg(channel, true);

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

