#include <aslam/common/thread-pool.h>

namespace aslam {

// the constructor just launches some amount of workers
ThreadPool::ThreadPool(size_t threads) : stop_(false) {
  for(size_t i = 0; i < threads; ++i)
    workers_.emplace_back( std::bind(&ThreadPool::run, this) );
}

// the destructor joins all threads
ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queueMutex_);
    stop_ = true;
  }
  condition_.notify_all();
  for(size_t i = 0; i < workers_.size(); ++i) {
    workers_[i].join();
  }
}

void ThreadPool::run() {
    while(true) {
      std::unique_lock<std::mutex> lock(this->queueMutex_);
      while(!this->stop_ && this->tasks_.empty()) {
        this->condition_.wait(lock);
      }
      if(this->stop_ && this->tasks_.empty()) {
        return;
      }
      std::function<void()> task(this->tasks_.front());
      this->tasks_.pop();
      lock.unlock();
      task();
      }
    }

}  // namespace aslam
