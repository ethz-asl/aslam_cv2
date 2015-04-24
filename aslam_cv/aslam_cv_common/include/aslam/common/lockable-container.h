#ifndef ASLAM_LOCKABLE_CONTAINER_H_
#define ASLAM_LOCKABLE_CONTAINER_H_

#include <memory>
#include <mutex>
#include <thread>

#include <Eigen/Core>
#include <glog/logging.h>

namespace aslam {
/// \class LockableContainer
template<typename DataType>
class LockableContainer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::unique_ptr<LockableContainer> UniquePtr;
  typedef std::shared_ptr<LockableContainer> Ptr;
  typedef std::shared_ptr<const LockableContainer> ConstPtr;

  typedef std::shared_ptr<DataType> DataTypePtr;
  typedef std::unique_ptr<DataType> DataTypeUniquePtr;

 protected:
  LockableContainer() {}
 public:
  LockableContainer(const LockableContainer&) = delete;
  void operator=(const LockableContainer&) = delete;
  virtual ~LockableContainer() {}

 public:
  // Factory function.
  template<typename ... Args>
  static std::shared_ptr<LockableContainer> create(Args&&... args) {
    LockableContainer::Ptr container(new LockableContainer);
    container->data_.reset(new DataType(args...));
    return container;
  }

  inline DataType* operator->() {
    CHECK(!m_data_.try_lock()) << "You must lock the container before accessing it.";
    CHECK(data_) << "The container does not contain a valid data object. Was it released?";
    return data_.get();
  }

  inline const DataType* operator->() const {
    CHECK(!m_data_.try_lock()) << "You must lock the container before accessing it.";
    CHECK(data_) << "The container does not contain a valid data object. Was it released?";
    return data_.get();
  }

  inline DataTypePtr release() {
    lock();
    CHECK(data_) << "The container does not contain a valid data object. Was it released?";
    DataTypePtr released_ptr(data_.release());
    unlock();
    return released_ptr;
  }

  inline void lock() const { m_data_.lock(); }
  inline void unlock() const { m_data_.unlock(); }

 public:
  DataTypeUniquePtr data_;
  mutable std::mutex m_data_;
};

}  // namespace aslam
#endif  // ASLAM_LOCKABLE_CONTAINER_H_
