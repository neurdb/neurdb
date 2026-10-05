/*
 *  Copyright (c) 2012-2019, Jyri J. Virkki
 *  All rights reserved.
 *
 *  This file is under BSD license. See LICENSE file.
 */

/*
 * Refer to bloom.h for documentation on the public interfaces.
 */
#include <setjmp.h>

#include "postgres.h"
#include <string.h>
#include "fmgr.h"
#include "utils/geo_decls.h"
#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "bloom.h"
#include "murmur2/murmurhash2.h"
#include <sys/mman.h>
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>

#define MAKESTRING(n) STRING(n)
#define STRING(n) #n

struct bloom * create_shared_bloom_ptr(key_t key){
    int shmid;
    struct bloom * mem;
    if ((shmid = shmget(key + BLOOM_PTR_SHARED_KEY_OFFSET,
                       sizeof(struct bloom), IPC_CREAT | IPC_EXCL | 0600)) < 0) {
        int old_shmid = shmget(key + BLOOM_PTR_SHARED_KEY_OFFSET,
                              sizeof(struct bloom), 0600);
        if (old_shmid >= 0)
            shmctl(old_shmid, IPC_RMID, NULL);
        shmid = shmget(key + BLOOM_PTR_SHARED_KEY_OFFSET,
                      sizeof(struct bloom), IPC_CREAT | IPC_EXCL | 0600);
        if (shmid < 0) {
            elog(ERROR, "could not create Bloom metadata shared memory: %m");
        }
    }
    if ((mem = shmat(shmid, NULL, 0)) == (void *) -1) {
        elog(ERROR, "could not attach Bloom metadata shared memory: %m");
    }
    memset(mem, 0, sizeof(struct bloom));
    return mem;
}

struct bloom * get_shared_bloom_ptr(key_t key){
    int shmid;
    struct bloom * mem;
    if ((shmid = shmget(key + BLOOM_PTR_SHARED_KEY_OFFSET,
                       sizeof(struct bloom), 0600)) < 0) {
        elog(ERROR, "could not find Bloom metadata shared memory: %m");
    }
    if ((mem = shmat(shmid, NULL, 0)) == (void *) -1) {
        elog(ERROR, "could not attach Bloom metadata shared memory: %m");
    }
    return mem;
}


unsigned char * create_bf_data_space(key_t key, int size){
    int shmid;
    unsigned char * mem;
    if ((shmid = shmget(key + BLOOM_DATA_SHARED_KEY_OFFSET, size,
                       IPC_CREAT | IPC_EXCL | 0600)) < 0) {
        int old_shmid = shmget(key + BLOOM_DATA_SHARED_KEY_OFFSET, size, 0600);
        if (old_shmid >= 0)
            shmctl(old_shmid, IPC_RMID, NULL);
        shmid = shmget(key + BLOOM_DATA_SHARED_KEY_OFFSET, size,
                      IPC_CREAT | IPC_EXCL | 0600);
        if (shmid < 0) {
            elog(ERROR, "could not create Bloom data shared memory: %m");
        }
    }
    if ((mem = shmat(shmid, NULL, 0)) == (void *) -1) {
        elog(ERROR, "could not attach Bloom data shared memory: %m");
    }
    memset(mem, 0, size);
    return mem;
}

unsigned char * get_bf_data_space(key_t key, int size){
    int shmid;
    unsigned char * mem;
    if ((shmid = shmget(key + BLOOM_DATA_SHARED_KEY_OFFSET, size, 0600)) < 0) {
        elog(ERROR, "could not find Bloom data shared memory: %m");
    }
    if ((mem = shmat(shmid, NULL, 0)) == (void *) -1) {
        elog(ERROR, "could not attach Bloom data shared memory: %m");
    }
    return mem;
}

inline static int test_bit_set_bit(unsigned char * buf,
                                   unsigned int x, int set_bit)
{
  unsigned int byte = x >> 3;
  unsigned char c = buf[byte];        // expensive memory access
  unsigned int mask = 1 << (x % 8);

  if (c & mask) {
    return 1;
  } else {
    if (set_bit) {
      buf[byte] = c | mask;
    }
    return 0;
  }
}


static int bloom_check_add(struct bloom * bloom,
                           const void * buffer, int len, int add)
{
  int hits = 0;
  register unsigned int a;
  register unsigned int b;
  register unsigned int x;
  register unsigned int i;

  if (bloom->ready == 0) {
    printf("bloom at %p not initialized!\n", (void *)bloom);
    return -1;
  }

  a = murmurhash2(buffer, len, 0x9747b28c);
  b = murmurhash2(buffer, len, a);

  for (i = 0; i < bloom->hashes; i++) {
    x = (a + i*b) % bloom->bits;
    if (test_bit_set_bit(bloom->bf, x, add)) {
      hits++;
    } else if (!add) {
      // Don't care about the presence of all the bits. Just our own.
      return 0;
    }
  }
  if (hits == bloom->hashes) {
    return 1;                // 1 == element already in (or collision)
  }
  return 0;
}

int bloom_init(struct bloom * bloom, int entries, double error, key_t key)
{
  double num;
  double denom;
  double dentries;

  bloom->ready = 0;
  bloom->add_count = 0;
  bloom->key = key;
  bloom->probe_cnt = 0;
  bloom->prune_cnt = 0;
  bloom->total_probe_cnt = 0;

  if (entries < 1000 || error == 0) {
    return 1;
  }

  bloom->entries = entries;
  bloom->error = error;

  num = log(bloom->error);
  denom = 0.480453013918201; // ln(2)^2
  bloom->bpe = -(num / denom);
  // elog(NOTICE, "Before init");

  dentries = (double)entries;
  bloom->bits = (int)(dentries * bloom->bpe);

  if (bloom->bits % 8) {
    bloom->bytes = (bloom->bits / 8) + 1;
  } else {
    bloom->bytes = bloom->bits / 8;
  }
  // elog(NOTICE, "Before init");

  bloom->hashes = (int)ceil(0.693147180559945 * bloom->bpe);  // ln(2)
  // elog(NOTICE, "Before init");

  // bloom->bf = (unsigned char *) create_bf_data_space(key, bloom->bytes * sizeof(unsigned char));

  // bloom->bf = (unsigned char *)create_shared_memory(bloom->bytes * sizeof(unsigned char));
  bloom->bf = (unsigned char *)calloc(bloom->bytes, sizeof(unsigned char));
  // use palloc rather than calloc
  // bloom->bf = (unsigned char *)palloc(bloom->bytes * sizeof(unsigned char));
  // bloom->bf = (unsigned char *)malloc(bloom->bytes * sizeof(unsigned char));

  if (bloom->bf == NULL) {                                   // LCOV_EXCL_START
    return 1;
  }                                                          // LCOV_EXCL_STOP

  bloom->ready = 1;
  return 0;
}


int bloom_check(struct bloom * bloom, const void * buffer, int len)
{
  return bloom_check_add(bloom, buffer, len, 0);
}


int bloom_add(struct bloom * bloom, const void * buffer, int len)
{
  bloom->add_count += 1;
  return bloom_check_add(bloom, buffer, len, 1);
}


void bloom_print(struct bloom * bloom)
{
  printf("bloom at %p\n", (void *)bloom);
  printf(" ->entries = %d\n", bloom->entries);
  printf(" ->error = %f\n", bloom->error);
  printf(" ->bits = %d\n", bloom->bits);
  printf(" ->bits per elem = %f\n", bloom->bpe);
  printf(" ->bytes = %d\n", bloom->bytes);
  printf(" ->hash functions = %d\n", bloom->hashes);
}

struct bloom * bloom_bit_and(struct bloom * bloom1, struct bloom * bloom2)
{
  int i = 0;
  assert(bloom1->bytes == bloom2->bytes);
  for (i = 0; i < bloom1->bytes; i++){
    bloom1->bf[i] = bloom1->bf[i] & bloom2->bf[i];
  }
  return bloom1;
}

struct bloom * bloom_bit_or(struct bloom * bloom1, struct bloom * bloom2)
{
  int i = 0;
  assert(bloom1->bytes == bloom2->bytes);
  for (i = 0; i < bloom1->bytes; i++){
    bloom1->bf[i] = bloom1->bf[i] | bloom2->bf[i];
  }
  return bloom1;
}

int bloom_get_content_signature(struct bloom * bloom)
{
  int sig = 0;
  int i = 0;
  for(i=0; i<bloom->bytes;i++){
    sig += (int) bloom->bf[i];
  }
  return sig;
}



void bloom_free(struct bloom * bloom)
{
  if (bloom->ready) {
    free(bloom->bf);
  }
  bloom->ready = 0;
}


int bloom_reset(struct bloom * bloom)
{
  if (!bloom->ready) return 1;
  memset(bloom->bf, 0, bloom->bytes);
  return 0;
}


const char * bloom_version(void)
{
  return MAKESTRING(BLOOM_VERSION);
}


struct bloom * bloom_cpy(
  struct bloom * new_bloom,
  unsigned char * new_bf_data,
  struct bloom * shared_bloom,
  unsigned char * shared_bf_data){
    new_bloom -> entries = shared_bloom -> entries;
    new_bloom -> error = shared_bloom -> error;
    new_bloom -> bits = shared_bloom -> bits;
    new_bloom -> bytes = shared_bloom -> bytes;
    new_bloom -> hashes = shared_bloom -> hashes;
    new_bloom -> bpe = shared_bloom -> bpe;
    new_bloom -> ready = shared_bloom -> ready;
    new_bloom -> add_count = shared_bloom -> add_count;
    new_bloom -> key = shared_bloom -> key;

    // int i = 0;
    // for (i=0; i<MAX_BLOOM_SIZE; i++){
    //   new_bf_data[i] = shared_bf_data[i];
    //   if (new_bf_data[i] != shared_bf_data[i]) {
    //     exit(1);
    //   }
    // }
    // new_bloom -> bf = new_bf_data;
    return new_bloom;
}
