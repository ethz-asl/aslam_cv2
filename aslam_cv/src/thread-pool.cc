#include <aslam/common/thread-pool.h>

namespace aslam {

// The constructor just launches some amount of workers.
ThreadPool::ThreadPool(size_t threads) : active_threads_(0), stop_(false) {
  for(size_t i = 0; i < threads; ++i)
    workers_.emplace_back(std::bind(&ThreadPool::run, this));
}

// The destructor joins all threads.
ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(tasks_mutex_);
    stop_ = true;
  }
  tasks_condition_.notify_all();
  for(size_t i = 0; i < workers_.size(); ++i) {
    workers_[i].join();
  }
}

void ThreadPool::run() {
    while(true) {
      std::unique_lock<std::mutex> lock(this->tasks_mutex_);
      while(!this->stop_ && this->tasks_.empty()) {
        this->tasks_condition_.wait(lock);
      }
      if(this->stop_ && this->tasks_.empty()) {
        return;
      }
      std::function<void()> task(this->tasks_.front());
      this->tasks_.pop();
      ++active_threads_;
      // Unlock the queue while we execute the task.
      lock.unlock();
      task();
      lock.lock();
      --active_threads_;
      // This is the secret to making the waitForEmptyQueue() function work.
      // After finishing a task, notify that this work is done.
      wait_condition_.notify_all();
    }
}

void ThreadPool::waitForEmptyQueue() const {
  std::unique_lock<std::mutex> lock(this->tasks_mutex_);
  // Only exit if all tasks are complete by tracking the number of
  // active threads.
  while(active_threads_ || !tasks_.empty() ) {
      this->wait_condition_.wait(lock);
  }
}
}  // namespace aslam
