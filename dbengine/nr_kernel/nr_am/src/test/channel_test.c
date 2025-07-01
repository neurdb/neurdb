#include "ipc/msg.h"
#include "nram_utils/config.h"
#include <sys/time.h>
#include <sys/wait.h>

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
        snprintf(msg, sizeof(msg), "msg_%05d-%04d", getpid(), produced);

        gettimeofday(&start, NULL);

        while (produced < total_messages) {
            if (KVChannelPush(channel, msg, strlen(msg) + 1, true)) {
                produced++;
            } else {
                pg_usleep(1000);  // Sleep 1ms to avoid tight loop
            }

            gettimeofday(&now, NULL);
            elapsed = (now.tv_sec - start.tv_sec) + (now.tv_usec - start.tv_usec) / 1e6;
            if (elapsed > timeout_secs) {
                PrintChannelContent(channel);
                elog(ERROR, "[Producer %d] Timeout after %.2f seconds", getpid(), timeout_secs);
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
            if (KVChannelPop(channel, buf, 16, false)) {
                if (strncmp(buf, "msg_", 4) != 0) {
                    PrintChannelContent(channel);
                    elog(ERROR, "[Consumer %d] Corrupted message: %s", getpid(), buf);
                }
                consumed++;
            } else {
                pg_usleep(1000);  // Sleep 1ms to reduce CPU usage
            }

            gettimeofday(&now, NULL);
            elapsed = (now.tv_sec - start.tv_sec) + (now.tv_usec - start.tv_usec) / 1e6;
            if (elapsed > timeout_secs) {
                PrintChannelContent(channel);
                elog(ERROR, "[Consumer %d] Timeout after %.2f seconds", getpid(), timeout_secs);
            }
        }
        _exit(0);
    }

    /* Parent waits */
    waitpid(producer_pid, &status, 0);
    Assert(status == 0);
    waitpid(consumer_pid, &status, 0);
    Assert(status == 0);

    KVChannelDestroy(channel);
    elog(INFO, "KV channel multi-process test passed");
}


void run_channel_msg_basic_test(void) {
    KVMsg msg = NewMsg(kv_put, 12345), *recv;
    char* data = "Hello, KV World!";
    KVChannel *channel = KVChannelInit("basic", true);
    
    msg.header.entitySize = strlen(data) + 1;
    msg.entity = data;
    msg.writer = DefaultWriteEntity;
    msg.reader = DefaultReadEntity;
    // PrintKVMsg(&msg);

    if (!KVChannelPushMsg(channel, &msg, false)) {
        elog(ERROR, "Message channel push fail");
    }
    // PrintChannelContent(channel);

    recv = palloc(sizeof(KVMsg));

    if(!KVChannelPopMsg(channel, recv, false)) {
        // PrintKVMsg(recv);
        elog(ERROR, "Message channel pop fail");
    }


    Assert(recv->header.op == kv_put);
    Assert(recv->header.relId == 12345);
    Assert(recv->header.entitySize == strlen(data) + 1);
    Assert(recv->entity != NULL);
    Assert(strcmp((char*)recv->entity, data) == 0);
    // Assert(false);

    pfree(recv->entity);
    elog(INFO, "KV channel multi-process test passed");
}



// /*
//  * This test:
//  * 1. Initializes KVChannel
//  * 2. Spawns producer and consumer processes
//  * 3. Pushes/Pops KVMsg with entity payload
//  * 4. Covers wrap-around edge case and correctness
//  */
// void run_msg_multiprocess_test(void) {
//     KVChannel* channel;
//     char* name = "kv_msg_test_channel";
//     pid_t producer_pid, consumer_pid;
//     int total_messages = 10, status = 0;
//     double timeout_secs = 2.0;

//     channel = KVChannelInit(name, true);

//     producer_pid = fork();
//     if (producer_pid == 0) {
//         /* Producer process */
//         int produced = 0;
//         struct timeval start, now;
//         double elapsed;

//         channel = KVChannelInit(name, false);
//         gettimeofday(&start, NULL);

//         while (produced < total_messages) {
//             KVMsg msg = NewMsg(kv_put, 1000 + produced);
//             char payload[64];

//             snprintf(payload, sizeof(payload), "payload_%d_pid_%d", produced, getpid());
//             msg.header.entitySize = strlen(payload) + 1;
//             msg.entity = payload;
//             msg.writer = DefaultWriteEntity;
//             msg.reader = DefaultReadEntity;

//             if (KVChannelPushMsg(channel, &msg, true)) {
//                 produced++;
//             } else {
//                 pg_usleep(1000);  // Sleep to avoid tight loop
//             }

//             gettimeofday(&now, NULL);
//             elapsed = (now.tv_sec - start.tv_sec) + (now.tv_usec - start.tv_usec) / 1e6;
//             if (elapsed > timeout_secs) {
//                 PrintChannelContent(channel);
//                 elog(ERROR, "[Producer %d] Timeout after %.2f seconds", getpid(), timeout_secs);
//             }
//         }
//         _exit(0);
//     }

//     consumer_pid = fork();
//     if (consumer_pid == 0) {
//         /* Consumer process */
//         int consumed = 0;
//         struct timeval start, now;
//         double elapsed;

//         channel = KVChannelInit(name, false);
//         gettimeofday(&start, NULL);

//         while (consumed < total_messages) {
//             KVMsg recv;
//             memset(&recv, 0, sizeof(KVMsg));

//             if (KVChannelPopMsg(channel, &recv, false)) {
//                 /* Validate */
//                 if (recv.header.op != kv_put || recv.header.relId != 1000 + consumed) {
//                     PrintChannelContent(channel);
//                     elog(ERROR, "[Consumer %d] Header mismatch: op=%d relId=%u",
//                          getpid(), recv.header.op, recv.header.relId);
//                 }

//                 if (recv.header.entitySize == 0 || recv.entity == NULL) {
//                     elog(ERROR, "[Consumer %d] Missing entity payload", getpid());
//                 }

//                 if (strncmp((char*)recv.entity, "payload_", 8) != 0) {
//                     PrintChannelContent(channel);
//                     elog(ERROR, "[Consumer %d] Corrupted entity: %s", getpid(), (char*)recv.entity);
//                 }

//                 pfree(recv.entity);
//                 consumed++;
//             } else {
//                 pg_usleep(1000);  // Avoid busy wait
//             }

//             gettimeofday(&now, NULL);
//             elapsed = (now.tv_sec - start.tv_sec) + (now.tv_usec - start.tv_usec) / 1e6;
//             if (elapsed > timeout_secs) {
//                 PrintChannelContent(channel);
//                 elog(ERROR, "[Consumer %d] Timeout after %.2f seconds", getpid(), timeout_secs);
//             }
//         }
//         _exit(0);
//     }

//     /* Parent waits */
//     waitpid(producer_pid, &status, 0);
//     Assert(status == 0);
//     waitpid(consumer_pid, &status, 0);
//     Assert(status == 0);

//     KVChannelDestroy(channel);
//     elog(INFO, "KVMsg multi-process test passed");
// }
