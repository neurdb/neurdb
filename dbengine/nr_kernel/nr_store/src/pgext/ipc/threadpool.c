#include "pgext/ipc/threadpool.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

static void *worker_thread(void *arg);

void threadpool_init(ThreadPool *pool, int num_threads, int omp_parallelism) {
    assert(pool != NULL);
    assert(num_threads > 0);

    memset(pool, 0, sizeof(ThreadPool));
    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->cond, NULL);

    pool->num_threads = num_threads;
    pool->threads = (pthread_t *)malloc(sizeof(pthread_t) * num_threads);
    pool->omp_parallelism = omp_parallelism;
    pool->shutdown = false;
    pool->task_head = NULL;
    pool->task_tail = NULL;

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }
}

void threadpool_destroy(ThreadPool *pool) {
    ThreadTask *task, *tmp;

    assert(pool != NULL);
    pthread_mutex_lock(&pool->lock);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->lock);

    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    free(pool->threads);

    task = pool->task_head;
    while (task) {
        tmp = task;
        task = task->next;
        free(tmp);
    }

    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->cond);
}

void threadpool_add_task(ThreadPool *pool, TaskFunction func, void *arg) {
    ThreadTask *task = (ThreadTask *)malloc(sizeof(ThreadTask));
    task->function = func;
    task->arg = arg;
    task->next = NULL;

    assert(pool != NULL);
    assert(func != NULL);

    pthread_mutex_lock(&pool->lock);

    if (pool->task_tail) {
        pool->task_tail->next = task;
    } else {
        pool->task_head = task;
    }
    pool->task_tail = task;

    pthread_cond_signal(&pool->cond);
    pthread_mutex_unlock(&pool->lock);
}

static void *worker_thread(void *arg) {
    ThreadPool *pool = (ThreadPool *)arg;
    ThreadTask *task;
    omp_set_num_threads(pool->omp_parallelism);

    while (true) {
        pthread_mutex_lock(&pool->lock);

        while (pool->task_head == NULL && !pool->shutdown) {
            // wait for tasks
            pthread_cond_wait(&pool->cond, &pool->lock);
        }

        if (pool->shutdown && pool->task_head == NULL) {
            // no more tasks and threadpool is shutting down
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        task = pool->task_head;
        if (task) {
            pool->task_head = task->next;
            if (pool->task_head == NULL)
                pool->task_tail = NULL;
        }

        pthread_mutex_unlock(&pool->lock);

        if (task) {
            task->function(task->arg);
            free(task);
        }
    }

    return NULL;
}
