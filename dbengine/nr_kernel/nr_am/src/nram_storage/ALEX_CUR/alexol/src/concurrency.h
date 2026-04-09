//
// Created by zhong on 2021/3/18.
//
#pragma once

#include <atomic>
#include <array>
#include <immintrin.h>
#include <sched.h>
#include "tbb/queuing_rw_mutex.h"
#include "tbb/spin_rw_mutex.h"
#include "tbb/spin_mutex.h"
#include "tbb/mutex.h"
#include "tbb/reader_writer_lock.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/enumerable_thread_specific.h"

namespace alexol {
  
  template<typename T>
  void atomic_add(std::atomic <T> &target, T operand) {
    while (true) {
      auto expected = target.load();
      auto desired = expected + operand;
      if (target.compare_exchange_strong(expected, desired)) {
        break;
      }
    }
  }

  class spin_lock {
  private:
    std::atomic_bool lock_;
      public:
        spin_lock() {
          lock_.store(false);
        }
        ~spin_lock() {
          lock_.store(false);
        }
        void lock(){
          while(lock_.exchange(true)){}
        }

        bool try_lock() {
          return !lock_.exchange(true);
        }

        void unlock(){
          while(!lock_.exchange(false)){}
        }

        bool test(){
          return lock_.load();
        }

        void wait(){
          while(lock_.load()){};
        }
  };

  void yield(int count) {
    if (count>3)
      sched_yield();
    else
      _mm_pause();
  }

  // optimistic lock implementation is based on https://github.com/wangziqi2016/index-microbench/blob/master/BTreeOLC/BTreeOLC_child_layout.h
  struct OptLock {
    std::atomic<uint64_t> typeVersionLockObsolete{0b100};

    OptLock() = default;
    OptLock(const OptLock& other) {
      typeVersionLockObsolete = 0b100;
    }

    uint64_t get_version_number()
    {
      return typeVersionLockObsolete.load();
    }

    bool isLocked(uint64_t version) {
      return ((version & 0b10) == 0b10);
    }

    bool isLocked() {
      return ((typeVersionLockObsolete.load() & 0b10) == 0b10);
    }

    uint64_t readLockOrRestart(bool &needRestart) {
      uint64_t version;
      version = typeVersionLockObsolete.load();
      if (isLocked(version) || isObsolete(version)) {
        _mm_pause();
        needRestart = true;
      }
      return version;
    }

    void writeLockOrRestart(bool &needRestart) {
      uint64_t version;
      version = readLockOrRestart(needRestart);
      if (needRestart) return;

      upgradeToWriteLockOrRestart(version, needRestart);
    }

    void upgradeToWriteLockOrRestart(uint64_t &version, bool &needRestart) {
      if (typeVersionLockObsolete.compare_exchange_strong(version, version + 0b10)) {
        version = version + 0b10;
      } else {
        _mm_pause();
        needRestart = true;
      }
    }

    void writeUnlock() {
      typeVersionLockObsolete.fetch_add(0b10);
    }

    bool isObsolete(uint64_t version) {
      return (version & 1) == 1;
    }

    bool isObsolete() {
      return (typeVersionLockObsolete.load() & 1) == 1;
    }

    void checkOrRestart(uint64_t startRead, bool &needRestart) const {
      readUnlockOrRestart(startRead, needRestart);
    }

    void readUnlockOrRestart(uint64_t startRead, bool &needRestart) const {
      needRestart = (startRead != typeVersionLockObsolete.load());
    }

    void writeUnlockObsolete() {
      typeVersionLockObsolete.fetch_add(0b11);
    }

    void labelObsolete() {
      typeVersionLockObsolete.store((typeVersionLockObsolete.load() | 1));
    }
  };
  typedef tbb::spin_rw_mutex Alex_rw_mutex;
  typedef OptLock Alex_mutex;
  typedef Alex_rw_mutex::scoped_lock Alex_rw_lock;
  typedef tbb::spin_mutex Alex_strict_mutex;
  typedef Alex_strict_mutex::scoped_lock Alex_strict_lock;
}