#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <aslam/common/lockable-container.h>

constexpr size_t kTestNumber = 10;
class TestObject {
 public:
  TestObject() : number_(kTestNumber) {}
  size_t number() { return number_; }
 private:
  size_t number_;
};

TEST(TestLockableContainer, BasicAccess) {
  // Create a container.
  typedef aslam::LockableContainer<TestObject> LockableObject;
  LockableObject::Ptr test_container = LockableObject::create();

  // Check death when accessing without locking.
  const std::string kLockedErrorMsg("You must lock the container before accessing it.");
  EXPECT_DEATH((*test_container)->number(), kLockedErrorMsg);

  // Check access with locking.
  test_container->lock();
  EXPECT_EQ((*test_container)->number(), kTestNumber);
}

TEST(TestLockableContainer, ReleaseObject) {
  // Create a container.
  typedef aslam::LockableContainer<TestObject> LockableObject;
  LockableObject::Ptr test_container = LockableObject::create();

  // Test releasing the contained object.
  std::shared_ptr<TestObject> object = test_container->release();
  ASSERT_TRUE(object.get() != nullptr);
  EXPECT_EQ(object->number(), kTestNumber);

  const std::string kRelasedErrorMsg(
      "The container does not contain a valid data object. Was it released?");
  // Check death on accessing an empty container.
  EXPECT_DEATH(test_container->lock(), kRelasedErrorMsg);

  // Check death on releasing an empty container.
  EXPECT_DEATH(test_container->release(), kRelasedErrorMsg);
  EXPECT_FALSE(test_container->isSet());
}

TEST(TestLockableContainer, CreateFromExistingObject) {
  std::shared_ptr<TestObject> existing_object(new TestObject);
  ASSERT_TRUE(existing_object.get() != nullptr);

  // Create a container.
  typedef aslam::LockableContainer<TestObject> LockableObject;
  LockableObject::Ptr test_container = LockableObject::createFromExistingObject(existing_object);
  ASSERT_TRUE(existing_object.get() == nullptr);

  // Check death when accessing without locking.
  const std::string kLockedErrorMsg("You must lock the container before accessing it.");
  EXPECT_DEATH((*test_container)->number(), kLockedErrorMsg);

  // Check access with locking.
  test_container->lock();
  EXPECT_EQ((*test_container)->number(), kTestNumber);
  test_container->unlock();

  // Test releasing the contained object.
  std::shared_ptr<TestObject> object = test_container->release();
  ASSERT_TRUE(object.get() != nullptr);
  EXPECT_FALSE(test_container->isSet());
  EXPECT_EQ(object->number(), kTestNumber);
}

TEST(TestLockableContainer, DeathCreateFromExistingObjectWithManyReferences) {
  std::shared_ptr<TestObject> existing_object(new TestObject);
  ASSERT_TRUE(existing_object.get() != nullptr);
  std::shared_ptr<TestObject> second_reference = existing_object;

  // Create a container.
  typedef aslam::LockableContainer<TestObject> LockableObject;
  EXPECT_DEATH(LockableObject::createFromExistingObject(existing_object),
               "Can only manage objects with a single reference count.");
}

ASLAM_UNITTEST_ENTRYPOINT
