#include "microservice/ns_micro_threadpool.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <omp.h>


static void *worker_thread(void *arg);

void micro_threadpool_init(MicroThreadPool *pool, int num_threads, int omp_parallelism) {
    assert(pool != NULL);
    assert(num_threads > 0);

    memset(pool, 0, sizeof(MicroThreadPool));
    pthread_mutex_init(&pool->lock, nullptr);
    pthread_cond_init(&pool->cond, nullptr);

    pool->num_threads = num_threads;
    pool->threads = static_cast<pthread_t *>(malloc(sizeof(pthread_t) * num_threads));
    pool->omp_parallelism.store(omp_parallelism > 0 ? omp_parallelism : 1);
    pool->shutdown = false;
    pool->task_head = nullptr;
    pool->task_tail = nullptr;

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], nullptr, worker_thread, pool);
    }
}

void micro_threadpool_destroy(MicroThreadPool *pool) {
    MicroThreadTask *task, *tmp;

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

void micro_threadpool_add_task(MicroThreadPool *pool, MicroTaskFunction func, void *arg) {
    auto task = static_cast<MicroThreadTask *>(malloc(sizeof(MicroThreadTask)));
    task->function = func;
    task->arg = arg;
    task->next = nullptr;

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
    auto pool = static_cast<MicroThreadPool *>(arg);
    MicroThreadTask *task;

    while (true) {
        pthread_mutex_lock(&pool->lock);

        while (pool->task_head == nullptr && !pool->shutdown) {
            // wait for tasks
            pthread_cond_wait(&pool->cond, &pool->lock);
        }

        if (pool->shutdown && pool->task_head == nullptr) {
            // no more tasks and threadpool is shutting down
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        task = pool->task_head;
        if (task) {
            pool->task_head = task->next;
            if (pool->task_head == nullptr)
                pool->task_tail = nullptr;
        }

        pthread_mutex_unlock(&pool->lock);

        if (task) {
            int omp_threads = pool->omp_parallelism.load(std::memory_order_relaxed);
            if (omp_get_max_threads() != omp_threads) {
                omp_set_num_threads(omp_threads);
            }
            task->function(task->arg);
            free(task);
        }
    }
    return nullptr;
}
