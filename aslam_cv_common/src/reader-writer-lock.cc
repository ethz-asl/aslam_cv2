#include "aslam/common/reader-writer-lock.h"

// Adapted from http://www.paulbridger.com/read_write_lock/
namespace aslam {

ReaderWriterMutex::ReaderWriterMutex()
    : num_readers_(0),
      num_pending_writers_(0),
      current_writer_(false),
      pending_upgrade_(false) {}

ReaderWriterMutex::~ReaderWriterMutex() {}

void ReaderWriterMutex::acquireReadLock() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (num_pending_writers_ != 0 || pending_upgrade_ || current_writer_) {
    m_writerFinished.wait(lock);
  }
  ++num_readers_;
}

void ReaderWriterMutex::releaseReadLock() {
  std::unique_lock<std::mutex> lock(mutex_);
  --num_readers_;
  if (num_readers_ == (pending_upgrade_ ? 1 : 0)) {
    cv_readers.notify_all();
  }
}

void ReaderWriterMutex::acquireWriteLock() {
  std::unique_lock<std::mutex> lock(mutex_);
  ++num_pending_writers_;
  while (num_readers_ > (pending_upgrade_ ? 1 : 0)) {
    cv_readers.wait(lock);
  }
  while (current_writer_ || pending_upgrade_) {
    m_writerFinished.wait(lock);
  }
  --num_pending_writers_;
  current_writer_ = true;
}

void ReaderWriterMutex::releaseWriteLock() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    current_writer_ = false;
  }
  m_writerFinished.notify_all();
}

// Attempt upgrade. If upgrade fails, relinquish read lock.
bool ReaderWriterMutex::upgradeToWriteLock() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (pending_upgrade_) {
    --num_readers_;
    if (num_readers_ == 1) {
      cv_readers.notify_all();
    }
    return false;
  }
  pending_upgrade_ = true;
  while (num_readers_ > 1) {
    cv_readers.wait(lock);
  }
  pending_upgrade_ = false;
  --num_readers_;
  current_writer_ = true;
  if (num_readers_ == 0) {
    cv_readers.notify_all();
  }
  return true;
}

ScopedReadLock::ScopedReadLock(ReaderWriterMutex* rw_lock) : rw_lock_(rw_lock), locked_(true) {
  CHECK_NOTNULL(rw_lock_)->acquireReadLock();
}

ScopedReadLock::ScopedReadLock(ScopedReadLock&& other)
    : rw_lock_(std::move(other.rw_lock_)), locked_(true) {
  other.locked_ = false;
}

ScopedReadLock::~ScopedReadLock() {
  if (locked_) {
    rw_lock_->releaseReadLock();
  }
}

ScopedWriteLock::ScopedWriteLock(ReaderWriterMutex* rw_lock) : rw_lock_(rw_lock), locked_(true) {
  CHECK_NOTNULL(rw_lock_)->acquireWriteLock();
}

ScopedWriteLock::ScopedWriteLock(ScopedWriteLock&& other)
    : rw_lock_(std::move(other.rw_lock_)), locked_(true) {
  other.locked_ = false;
}

ScopedWriteLock::~ScopedWriteLock() {
  if (locked_) {
    rw_lock_->releaseWriteLock();
  }
}

}  // namespace aslam
