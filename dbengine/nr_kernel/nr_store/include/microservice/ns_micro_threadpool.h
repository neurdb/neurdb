#ifndef NS_THREADPOOL_H
#define NS_THREADPOOL_H

#include <atomic>
#include <pthread.h>
#include <stdbool.h>


typedef void *(*MicroTaskFunction)(void *arg);

typedef struct MicroThreadTask {
    MicroTaskFunction function;
    void *arg;
    struct MicroThreadTask *next;
} MicroThreadTask;

typedef struct MicroThreadPool {
    pthread_mutex_t lock;
    pthread_cond_t cond;

    pthread_t *threads;
    int num_threads;
    std::atomic<int> omp_parallelism;
    bool shutdown;

    MicroThreadTask *task_head;
    MicroThreadTask *task_tail;
} MicroThreadPool;

void micro_threadpool_init(MicroThreadPool *pool, int num_threads, int omp_parallelism);

void micro_threadpool_destroy(MicroThreadPool *pool);

void micro_threadpool_add_task(MicroThreadPool *pool, MicroTaskFunction func, void *arg);

#endif //NS_THREADPOOL_H
