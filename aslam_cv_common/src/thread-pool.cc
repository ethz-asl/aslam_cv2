#include <algorithm>

#include <aslam/common/thread-pool.h>

namespace aslam {

// The constructor just launches some amount of workers.
ThreadPool::ThreadPool(size_t threads)
    : active_threads_(0),
      stop_(false) {
  for (size_t i = 0; i < threads; ++i)
    workers_.emplace_back(std::bind(&ThreadPool::run, this));
}

// The destructor joins all threads.
ThreadPool::~ThreadPool() {
  {
    std::unique_lock < std::mutex > lock(tasks_mutex_);
    stop_ = true;
  }
  tasks_queue_change_.notify_all();
  for (size_t i = 0; i < workers_.size(); ++i) {
    workers_[i].join();
  }
}

void ThreadPool::run() {
  while (true) {
    std::unique_lock<std::mutex> lock(this->tasks_mutex_);

    // Here we need to select the next task from from a queue that is not
    // already serviced.
    std::function<void()> task;
    int group_id_of_task = -2;
    while (true) {
      const bool all_guards_active = std::all_of(
          task_group_exclusivity_guards_.begin(),
          task_group_exclusivity_guards_.end(),
          [](const std::pair<const int, bool>& value) {
            return value.second;
          });

      if (!all_guards_active || num_queued_nonexclusive_tasks > 0u) {
        // If all guards are active, we can go back to sleep until a thread
        // reports back for work.
        size_t index = 0u;
        for (const std::pair<const int, std::function<void()>>& groupid_task :
            groupid_tasks_) {
          // We have found a task to process if no thread is already working on
          // this group id.
          const bool is_exclusive_task =
              groupid_task.first != kGroupdIdNonExclusiveTask;
          bool guard_active = false;
          if (is_exclusive_task) {
            guard_active = task_group_exclusivity_guards_[groupid_task.first];
          }

          if (!(is_exclusive_task && guard_active)) {
            group_id_of_task = groupid_task.first;
            task = groupid_task.second;

            groupid_tasks_.erase(groupid_tasks_.begin() + index);
            if (!is_exclusive_task) {
              --num_queued_nonexclusive_tasks;
            }

            // We jump out of the nested for-structure here, because we have
            // found a task to process.
            goto task_found;
          }
          ++index;
        }
      }

      // Wait until the queue has changed (addition/removal) before re-checking
      // for new tasks to process.
      if (this->stop_ && groupid_tasks_.size() == 0u) {
        return;
      }

      this->tasks_queue_change_.wait(lock);
    }

    // We jump here if we found a task.
    task_found:
    CHECK(task);
    CHECK_GE(group_id_of_task, kGroupdIdNonExclusiveTask);

    ++active_threads_;

    // Make sure the no other thread is currently working on this exclusivity
    // group.
    if (group_id_of_task != kGroupdIdNonExclusiveTask) {
      std::unordered_map<size_t, bool>::iterator it_group_id_servied =
          task_group_exclusivity_guards_.find(group_id_of_task);
      CHECK(it_group_id_servied == task_group_exclusivity_guards_.end() ||
            it_group_id_servied->second == false);
      it_group_id_servied->second = true;
    }

    // Unlock the queue while we execute the task.
    lock.unlock();
    task();
    lock.lock();

    // Release the group for other threads.
    if (group_id_of_task != kGroupdIdNonExclusiveTask) {
      std::unordered_map<size_t, bool>::iterator it_group_id_servied =
          task_group_exclusivity_guards_.find(group_id_of_task);
      CHECK(it_group_id_servied != task_group_exclusivity_guards_.end() &&
            it_group_id_servied->second == true);
      it_group_id_servied->second = false;
    }

    --active_threads_;

    // This is the secret to making the waitForEmptyQueue() function work.
    // After finishing a task, notify that this work is done.
    tasks_queue_change_.notify_all();
  }
}

void ThreadPool::waitForEmptyQueue() const {
  std::unique_lock<std::mutex> lock(this->tasks_mutex_);
  // Only exit if all tasks are complete by tracking the number of
  // active threads.
  while (active_threads_ > 0u || groupid_tasks_.size() > 0u) {
    this->tasks_queue_change_.wait(lock);
  }
}
}  // namespace aslam
