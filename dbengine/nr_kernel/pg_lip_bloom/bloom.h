/*
 *  Copyright (c) 2012-2017, Jyri J. Virkki
 *  All rights reserved.
 *
 *  This file is under BSD license. See LICENSE file.
 */

#ifndef _BLOOM_H
#define _BLOOM_H

#ifdef __cplusplus
extern "C" {
#endif

#define BLOOM_PTR_SHARED_KEY_OFFSET 1234
#define BLOOM_DATA_SHARED_KEY_OFFSET 2345
#define MAX_BLOOM_FILTERS 10
#define MAX_BLOOM_SIZE 10000000
// #define LARGE_BLOOM_SIZE 10000000000

/** ***************************************************************************
 * Structure to keep track of one bloom filter.  Caller needs to
 * allocate this and pass it to the functions below. First call for
 * every struct must be to bloom_init().
 *
 */
struct bloom
{
  // These fields are part of the public interface of this structure.
  // Client code may read these values if desired. Client code MUST NOT
  // modify any of these.
  int entries;
  double error;
  int bits;
  int bytes;
  int hashes;

  // Fields below are private to the implementation. These may go away or
  // change incompatibly at any moment. Client code MUST NOT access or rely
  // on these.
  double bpe;
  unsigned char * bf;
  int ready;
  int add_count;
  key_t key;
  float probe_cnt;
  float prune_cnt;
  float stale_timer;
  int32 total_probe_cnt;
  // int * probed_keys;
};


/** ***************************************************************************
 * Initialize the bloom filter for use.
 *
 * The filter is initialized with a bit field and number of hash functions
 * according to the computations from the wikipedia entry:
 *     http://en.wikipedia.org/wiki/Bloom_filter
 *
 * Optimal number of bits is:
 *     bits = (entries * ln(error)) / ln(2)^2
 *
 * Optimal number of hash functions is:
 *     hashes = bpe * ln(2)
 *
 * Parameters:
 * -----------
 *     bloom   - Pointer to an allocated struct bloom (see above).
 *     entries - The expected number of entries which will be inserted.
 *               Must be at least 1000 (in practice, likely much larger).
 *     error   - Probability of collision (as long as entries are not
 *               exceeded).
 *
 * Return:
 * -------
 *     0 - on success
 *     1 - on failure
 *
 */
int bloom_init(struct bloom * bloom, int entries, double error, key_t key);


/** ***************************************************************************
 * Check if the given element is in the bloom filter. Remember this may
 * return false positive if a collision occurred.
 *
 * Parameters:
 * -----------
 *     bloom  - Pointer to an allocated struct bloom (see above).
 *     buffer - Pointer to buffer containing element to check.
 *     len    - Size of 'buffer'.
 *
 * Return:
 * -------
 *     0 - element is not present
 *     1 - element is present (or false positive due to collision)
 *    -1 - bloom not initialized
 *
 */
int bloom_check(struct bloom * bloom, const void * buffer, int len);


/** ***************************************************************************
 * Add the given element to the bloom filter.
 * The return code indicates if the element (or a collision) was already in,
 * so for the common check+add use case, no need to call check separately.
 *
 * Parameters:
 * -----------
 *     bloom  - Pointer to an allocated struct bloom (see above).
 *     buffer - Pointer to buffer containing element to add.
 *     len    - Size of 'buffer'.
 *
 * Return:
 * -------
 *     0 - element was not present and was added
 *     1 - element (or a collision) had already been added previously
 *    -1 - bloom not initialized
 *
 */
int bloom_add(struct bloom * bloom, const void * buffer, int len);


/** ***************************************************************************
 * Print (to stdout) info about this bloom filter. Debugging aid.
 *
 */
void bloom_print(struct bloom * bloom);

/** ***************************************************************************
 * Print (to stdout) info about this bloom filter. Debugging aid.
 *
 */
struct bloom * bloom_bit_and(struct bloom * bloom1, struct bloom * bloom2);

/** ***************************************************************************
 * Print (to stdout) info about this bloom filter. Debugging aid.
 *
 */
struct bloom * bloom_bit_or(struct bloom * bloom1, struct bloom * bloom2);



/** ***************************************************************************
 * Deallocate internal storage.
 *
 * Upon return, the bloom struct is no longer usable. You may call bloom_init
 * again on the same struct to reinitialize it again.
 *
 * Parameters:
 * -----------
 *     bloom  - Pointer to an allocated struct bloom (see above).
 *
 * Return: none
 *
 */
void bloom_free(struct bloom * bloom);

/** ***************************************************************************
 * Erase internal storage.
 *
 * Erases all elements. Upon return, the bloom struct returns to its initial
 * (initialized) state.
 *
 * Parameters:
 * -----------
 *     bloom  - Pointer to an allocated struct bloom (see above).
 *
 * Return:
 *     0 - on success
 *     1 - on failure
 *
 */
int bloom_reset(struct bloom * bloom);
struct bloom * create_shared_bloom_ptr(key_t key);
struct bloom * get_shared_bloom_ptr(key_t key);
unsigned char * create_bf_data_space(key_t key, int size);
unsigned char * get_bf_data_space(key_t key, int size);
int bloom_get_content_signature(struct bloom * bloom);

struct bloom * bloom_cpy(
  struct bloom * new_bloom,
  unsigned char * new_bf_data,
  struct bloom * shared_bloom,
  unsigned char * shared_bf_data);


/** ***************************************************************************
 * Returns version string compiled into library.
 *
 * Return: version string
 *
 */
const char * bloom_version(void);

#ifdef __cplusplus
}
#endif

#endif
