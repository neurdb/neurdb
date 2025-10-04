#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <pthread.h>
#include <stdbool.h>


typedef void *(*TaskFunction)(void *arg);

typedef struct ThreadTask {
    TaskFunction function;
    void *arg;
    struct ThreadTask *next;
} ThreadTask;

typedef struct ThreadPool {
    pthread_mutex_t lock;
    pthread_cond_t cond;

    pthread_t *threads;
    int num_threads;
    int omp_parallelism;
    bool shutdown;

    ThreadTask *task_head;
    ThreadTask *task_tail;
} ThreadPool;

void threadpool_init(ThreadPool *pool, int num_threads, int omp_parallelism);
void threadpool_destroy(ThreadPool *pool);
void threadpool_add_task(ThreadPool *pool, TaskFunction func, void *arg);

#endif //THREADPOOL_H
