#include "queue.h"

#include <stdlib.h>
#include <string.h>

void init_batch_queue(BatchQueue *queue, size_t max_size) {
  queue->head = NULL;
  queue->tail = NULL;
  queue->size = 0;
  queue->max_size = max_size;
  pthread_mutex_init(&queue->mutex, NULL);
  pthread_cond_init(&queue->consume, NULL);
  pthread_cond_init(&queue->produce, NULL);
}

void destroy_batch_queue(BatchQueue *queue) {
  pthread_mutex_destroy(&queue->mutex);
  pthread_cond_destroy(&queue->consume);
  pthread_cond_destroy(&queue->produce);
}

void enqueue(BatchQueue *queue, const char *batch_data) {
  BatchDataNode *node = (BatchDataNode *)malloc(sizeof(BatchDataNode));
  node->batched_data = strdup(batch_data);
  node->next = NULL;

  pthread_mutex_lock(&queue->mutex);  // lock the mutex
  while (queue->size >= queue->max_size) {
    // wait if the queue is full
    pthread_cond_wait(&queue->produce, &queue->mutex);
  }
  // LOCKED
  if (queue->tail) {
    queue->tail->next = node;
  } else {
    queue->head = node;
  }
  queue->tail = node;
  queue->size++;
  pthread_cond_signal(&queue->consume);  // signal the consumer
  pthread_mutex_unlock(&queue->mutex);   // unlock the mutex
}

char *dequeue(BatchQueue *queue) {
  pthread_mutex_lock(&queue->mutex);  // lock the mutex
  while (queue->head == NULL) {
    // wait if the queue is empty
    pthread_cond_wait(&queue->consume, &queue->mutex);
  }
  // LOCKED
  BatchDataNode *node = queue->head;
  queue->head = node->next;
  if (queue->head == NULL) {
    queue->tail = NULL;
  }
  queue->size--;
  pthread_cond_signal(&queue->produce);  // signal the producer
  pthread_mutex_unlock(&queue->mutex);   // unlock the mutex
  // extract the data
  char *data = node->batched_data;
  free(node);
  return data;
}
