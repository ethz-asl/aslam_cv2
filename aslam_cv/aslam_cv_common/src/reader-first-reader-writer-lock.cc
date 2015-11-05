#include "aslam/common/reader-first-reader-writer-lock.h"

namespace aslam {

ReaderFirstReaderWriterMutex::ReaderFirstReaderWriterMutex()
    : ReaderWriterMutex() {}

ReaderFirstReaderWriterMutex::~ReaderFirstReaderWriterMutex() {}

void ReaderFirstReaderWriterMutex::acquireReadLock() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (current_writer_) {
    m_writerFinished.wait(lock);
  }
  ++num_readers_;
}

void ReaderFirstReaderWriterMutex::acquireWriteLock() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    while (num_readers_ > (pending_upgrade_ ? 1 : 0)) {
      cv_readers.wait(lock);
    }
    if (!current_writer_ && !pending_upgrade_) {
      break;
    } else {
      while (current_writer_ || pending_upgrade_) {
        m_writerFinished.wait(lock);
      }
    }
  }
  current_writer_ = true;
}

}  // namespace aslam
