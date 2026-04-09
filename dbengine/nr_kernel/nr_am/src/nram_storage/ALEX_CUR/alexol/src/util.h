// Copyright (c) Simon Fraser University & The Chinese University of Hong Kong. All rights reserved.
// Licensed under the MIT license.
#pragma once

#include <immintrin.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <stdint.h>

#include <cstdint>
#include <iostream>

namespace alexol {

static constexpr const uint32_t kCacheLineSize = 64;

static bool FileExists(const char *pool_path) {
  struct stat buffer;
  return (stat(pool_path, &buffer) == 0);
}

#define LOG_FATAL(msg)      \
  std::cout << msg << "\n"; \
  exit(-1)

#define LOG(msg) std::cout << msg << "\n"

#define CAS(_p, _u, _v)                                             \
  (__atomic_compare_exchange_n(_p, _u, _v, false, __ATOMIC_ACQUIRE, \
                               __ATOMIC_ACQUIRE))

// ADD and SUB return the value after add or sub
#define ADD(_p, _v) (__atomic_add_fetch(_p, _v, __ATOMIC_ACQUIRE))
#define SUB(_p, _v) (__atomic_sub_fetch(_p, _v, __ATOMIC_ACQUIRE))
#define LOAD(_p) (__atomic_load_n(_p, __ATOMIC_SEQ_CST))
#define STORE(_p, _v) (__atomic_store_n(_p, _v, __ATOMIC_RELEASE))

#define SIMD 1
#define SIMD_CMP8(src, key)                                         \
  do {                                                              \
    const __m256i key_data = _mm256_set1_epi8(key);                 \
    __m256i seg_data =                                              \
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src)); \
    __m256i rv_mask = _mm256_cmpeq_epi8(seg_data, key_data);        \
    mask = _mm256_movemask_epi8(rv_mask);                           \
  } while (0)

#define SSE_CMP8(src, key)                                       \
  do {                                                           \
    const __m128i key_data = _mm_set1_epi8(key);                 \
    __m128i seg_data =                                           \
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src)); \
    __m128i rv_mask = _mm_cmpeq_epi8(seg_data, key_data);        \
    mask = _mm_movemask_epi8(rv_mask);                           \
  } while (0)

#define CHECK_BIT(var, pos) ((((var) & (1 << pos)) > 0) ? (1) : (0))

#define LOG2(X) (32 - __builtin_clz((X)) - 1)

#define CACHE_LINE_SIZE    64

inline void mfence(void) { asm volatile("mfence" ::: "memory"); }

int msleep(uint64_t msec) {
  struct timespec ts;
  int res;

  ts.tv_sec = msec / 1000;
  ts.tv_nsec = (msec % 1000) * 1000000;

  do {
    res = nanosleep(&ts, &ts);
  } while (res && errno == EINTR);

  return res;
}

template<class T>
inline bool cas_multiple_type(T *src, T* old_src, T new_value){
  uint64_t* uint_src = reinterpret_cast<uint64_t*>(src);
  uint64_t* uint_old_src = reinterpret_cast<uint64_t*>(old_src);
  uint64_t uint_new_value = *(reinterpret_cast<uint64_t*>(&new_value));
  return CAS(uint_src, uint_old_src, uint_new_value);
}

template<class T>
inline T load_multiple_type(T *src){
  uint64_t* uint_src = reinterpret_cast<uint64_t*>(src);
  uint64_t value = __atomic_load_n(uint_src, __ATOMIC_ACQUIRE);
  return *(reinterpret_cast<T*>(&value));
}

// obtain the starting address of a cache line
#define GET_LINE(addr) \
(((unsigned long long)(addr)) & (~(unsigned long long)(CACHE_LINE_SIZE-1)))

// check if address is aligned at line boundary
#define  Isaligned_Atline(addr) \
(!(((unsigned long long)(addr)) & (unsigned long long)(CACHE_LINE_SIZE-1)))

// Cacheline flush code, from shimin chen
// use clwb and sfence

/**
 * flush a cache line
 *
 * @param addr   the address of the cache line
 */
static inline
void clwb(void * addr)
{ asm volatile("clwb %0": :"m"(*((char *)addr))); } 

/**
 * flush [start, end]
 *
 * there are at most two lines.
 */
static inline
void clwb2(void *start, void *end)
{
  clwb(start);
  if (GET_LINE(start) != GET_LINE(end)) {
    clwb(end);
  }
}

/**
 * flush [start, end]
 *
 * there can be 1 to many lines
 */
static inline
void clwbmore(void *start, void *end)
{ 
  unsigned long long start_line= GET_LINE(start);
  unsigned long long end_line= GET_LINE(end);
  do {
    clwb((char *)start_line);
    start_line += CACHE_LINE_SIZE;
  } while (start_line <= end_line);
}

/**
 * call sfence
 */
static inline
void sfence(void)
{ asm volatile("sfence"); }

}