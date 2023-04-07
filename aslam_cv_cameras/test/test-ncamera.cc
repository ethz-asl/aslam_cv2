#include <Eigen/Core>
#include <eigen-checks/gtest.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/common/entrypoint.h>
#include <aslam/common/yaml-serialization.h>

TEST(TestNCameraYamlSerialization, testEmptyYaml) {
  YAML::Node node = YAML::Load("{}");
  aslam::NCamera::Ptr ncamera = aligned_shared<aslam::NCamera>();
  bool success = ncamera->deserialize(node);
  EXPECT_EQ(success, false);
}

TEST(TestNCameraYamlSerialization, testSerialization) {
  aslam::SensorId base_sensor_id;
  aslam::generateId(&base_sensor_id);
  aslam::NCamera::Ptr ncamera = 
      aslam::NCamera::createTestNCamera(4u, base_sensor_id);
  ASSERT_TRUE(ncamera.get() != nullptr);

  std::string filename = "test_ncamera.yaml";
  ncamera->serializeToFile(filename);

  aslam::NCamera::Ptr ncamera_loaded(new aslam::NCamera());
  ASSERT_TRUE(ncamera_loaded.get() != nullptr);
  ASSERT_TRUE(ncamera_loaded->deserializeFromFile(filename));
  ASSERT_TRUE(ncamera_loaded.get() != nullptr);

  EXPECT_EQ(ncamera_loaded->getId(), ncamera->getId());
  EXPECT_EQ(ncamera_loaded->getTopic(), ncamera->getTopic());
  EXPECT_EQ(ncamera_loaded->getDescription(), ncamera->getDescription());

  size_t num_cameras = ncamera_loaded->getNumCameras();
  EXPECT_EQ(num_cameras, 4u);

  for (size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
    const aslam::Camera& camera_loaded = ncamera_loaded->getCamera(cam_idx);
    const aslam::Camera& camera_gt = ncamera->getCamera(cam_idx);

    EXPECT_EQ(camera_loaded.getId(), camera_gt.getId());
    EXPECT_EQ(camera_loaded.getTopic(), camera_gt.getTopic());
    EXPECT_EQ(camera_loaded.getDescription(), camera_gt.getDescription());
    EXPECT_EQ(camera_loaded.imageHeight(), camera_gt.imageHeight());
    EXPECT_EQ(camera_loaded.imageWidth(), camera_gt.imageWidth());
    EXPECT_TRUE(EIGEN_MATRIX_NEAR(ncamera->get_T_B_C(cam_idx).getTransformationMatrix(),
                                  ncamera_loaded->get_T_B_C(cam_idx).getTransformationMatrix(),
                                  1e-8));
    EXPECT_TRUE(camera_loaded == camera_gt);
  }
  EXPECT_TRUE(*ncamera == *ncamera_loaded);
}

TEST(TestNCamera, testClone) {
  aslam::SensorId base_sensor_id;
  aslam::generateId(&base_sensor_id);
  aslam::NCamera::Ptr ncamera = 
      aslam::NCamera::createTestNCamera(4u, base_sensor_id);
  ASSERT_TRUE(ncamera.get() != nullptr);

  aslam::NCamera::Ptr ncamera_clone(ncamera->clone());
  EXPECT_TRUE(ncamera_clone.get() != ncamera.get());

  // Make sure all members are equal.
  EXPECT_EQ(ncamera->getId(), ncamera_clone->getId());
  EXPECT_EQ(ncamera->getTopic(), ncamera_clone->getTopic());
  EXPECT_EQ(ncamera->getDescription(), ncamera_clone->getDescription());
  EXPECT_EQ(ncamera->getNumCameras(), ncamera_clone->getNumCameras());

  for (int idx = 0; idx < static_cast<int>(ncamera->getNumCameras()); ++idx) {
    EXPECT_EQ(ncamera->get_T_B_C(idx), ncamera_clone->get_T_B_C(idx));
    // Make sure all individual camera objects are cloned and equal.
    EXPECT_TRUE(ncamera->getCamera(idx) == ncamera_clone->getCamera(idx));
    EXPECT_TRUE(ncamera->getCameraShared(idx).get() != ncamera_clone->getCameraShared(idx).get());

    // Make sure the id-idx mapping is preserved.
    EXPECT_EQ(ncamera->getCameraIndex(ncamera_clone->getCameraId(idx)), idx);
  }
}

ASLAM_UNITTEST_ENTRYPOINT
