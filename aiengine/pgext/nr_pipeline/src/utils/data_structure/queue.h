#ifndef QUEUE_H
#define QUEUE_H

#include <stddef.h>
#include <pthread.h>

/**
 * Batch data node
 */
typedef struct BatchDataNode {
  char *batched_data;
  struct BatchDataNode *next;
} BatchDataNode;

/**
 * Batch data queue
 */
typedef struct BatchDataQueue {
  BatchDataNode *head;
  BatchDataNode *tail;
  size_t size;
  size_t max_size;
  pthread_mutex_t mutex;
  pthread_cond_t
      consume;  // condition variable for the consumer - websocket thread
  pthread_cond_t produce;  // condition variable for the producer - main thread
} BatchQueue;

/**
 * Initialize a batch data queue
 * @param queue The batch data queue
 * @param max_size The maximum size of the queue
 */
void init_batch_queue(BatchQueue *queue, size_t max_size);

/**
 * Destroy a batch data queue
 * @param queue The batch data queue
 */
void destroy_batch_queue(BatchQueue *queue);

/**
 * Enqueue a batch data to the queue
 * @param queue The batch data queue
 * @param batch_data The batch data
 */
void enqueue(BatchQueue *queue, const char *batch_data);

/**
 * Dequeue a batch data from the queue
 * @param queue The batch data queue
 * @return batch_data The batch data
 */
char *dequeue(BatchQueue *queue);

#endif  // QUEUE_H
