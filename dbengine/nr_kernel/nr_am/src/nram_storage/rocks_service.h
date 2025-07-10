#ifndef ROCKS_SERVICE_H
#define ROCKS_SERVICE_H

#include "nram_storage/rocksengine.h"
#include "ipc/msg.h"
#include "nram_utils/config.h"
#include "postgres.h"
#include <pthread.h>
#include <stdbool.h>

#define ROCKS_RUNNING_BIT 1

/* ------------------------------------------------------------------------
 * Async response queue. We do not set size limit to this response queue since there is at most NumProc items.
 * ------------------------------------------------------------------------
 */

#define RESP_CACHE_CAPACITY


typedef struct ResultNode {
    KVMsg *msg;
    struct ResultNode *next;
} ResultNode;

typedef struct {
    pthread_mutex_t lock;
    ResultNode *head;
    ResultNode *tail;
    bool shutdown;
} ResultQueue;

void ResultQueueInit(ResultQueue *q);
void ResultQueueDestroy(ResultQueue *q);
void ResultQueuePush(ResultQueue *q, KVMsg *msg);
KVMsg *ResultQueuePop(ResultQueue *q);
bool ResultQueueIsEmpty(ResultQueue *q);
void PrintResultQueue(ResultQueue *q);

/* ------------------------------------------------------------------------
 * Server handlers.
 * ------------------------------------------------------------------------
 */

void run_rocks(int num_threads);
void run_rocks_no_thread(void);

void *process_request(void *arg);

KVMsg *handle_kv_get(KVMsg *msg);
KVMsg *handle_kv_put(KVMsg *msg);
KVMsg *handle_kv_range_scan(KVMsg *msg);

void nram_rocks_service_init(void);
void nram_rocks_service_terminate(void);

PGDLLEXPORT void rocks_service_main(Datum arg);

#endif // ROCKS_SERVICE_H
