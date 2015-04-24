#ifndef ASLAM_LOCKABLE_CONTAINER_H_
#define ASLAM_LOCKABLE_CONTAINER_H_

#include <memory>
#include <mutex>
#include <thread>

#include <Eigen/Core>
#include <glog/logging.h>

#include <aslam/common/macros.h>

namespace aslam {
/// \class LockableContainer
/// \brief Wraps an existing type in a container that provides basic lock functionality for
///        synchronized access to the managed object.
/// \code    class TestObject {
///           public:
///            TestObject(size_t number) : number_(number) {}
///            size_t number() { return number_; }
///           private:
///            size_t number_;
///          };
///          // Create a container.
///          typedef aslam::LockableContainer<TestObject> LockableObject;
///          LockableObject::Ptr test_container = LockableObject::create(12);
///
///          // Synchronized access.
///          test_container->lock();
///          std::cout << (*test_container)->number();
///          test_container->unlock();
///
///          // Releasing the object from the container.
///          std::shared_ptr<TestObject> object = test_container->release();
///          // test_container is now invalid.
template<typename DataType>
class LockableContainer {
 public:
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(LockableContainer);
  ASLAM_POINTER_TYPEDEFS(LockableContainer);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<DataType> DataTypePtr;
  typedef std::unique_ptr<DataType> DataTypeUniquePtr;

 protected:
  LockableContainer() {}

 public:
  virtual ~LockableContainer() {}

  template<typename ... Args>
  static std::shared_ptr<LockableContainer> create(Args&&... args) {
    LockableContainer::Ptr container(new LockableContainer);
    container->data_.reset(new DataType(args...));
    return container;
  }

  static std::shared_ptr<LockableContainer> createFromExistingObject(
      DataTypePtr& existing_object) {
    LockableContainer::Ptr container(new LockableContainer);
    CHECK(existing_object.unique()) << "Can only manage objects with a single reference count.";
    container->data_.swap(existing_object);
    return container;
  }

  inline DataType* operator->() {
    assertWeOwnLock();
    assertIsSet();
    return data_.get();
  }

  inline const DataType* operator->() const {
    assertWeOwnLock();
    assertIsSet();
    return data_.get();
  }

  inline DataTypePtr release() {
    lock();
    assertIsSet();
    DataTypePtr released_ptr = data_;
    data_.reset();
    unlock();
    return released_ptr;
  }

  inline void lock() const {
    assertIsSet();
    m_data_.lock();
  }

  inline void unlock() const {
    m_data_.unlock();
  }

  inline bool isSet() const {
    return static_cast<bool>(data_);
  }

 private:
  inline void assertWeOwnLock() const {
    CHECK(!m_data_.try_lock()) << "You must lock the container before accessing it.";
  }

  inline void assertIsSet() const {
    CHECK(isSet()) << "The container does not contain a valid data object. Was it released?";
  }

 public:
  DataTypePtr data_;
  mutable std::mutex m_data_;
};

#define ASLAM_DEFINE_LOCKABLE(TypeName)                \
  typedef aslam::LockableContainer<TypeName> Lockable##TypeName;

}  // namespace aslam
#endif  // ASLAM_LOCKABLE_CONTAINER_H_
