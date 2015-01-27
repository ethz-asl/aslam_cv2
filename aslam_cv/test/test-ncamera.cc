#include <Eigen/Core>
#include <eigen-checks/gtest.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/cameras/yaml/ncamera-yaml-serialization.h>
#include <aslam/common/entrypoint.h>
#include <aslam/common/yaml-serialization.h>

TEST(TestNCameraYamlSerialization, testEmptyYaml) {
  YAML::Node node = YAML::Load("{}");
  aslam::NCamera::Ptr ncamera;
  ncamera = node.as<aslam::NCamera::Ptr>();
  EXPECT_EQ(ncamera, nullptr);
}

TEST(TestNCameraYamlSerialization, testSerialization) {
  aslam::NCamera::Ptr ncamera = aslam::NCamera::createTestNCamera(4u);
  aslam::NCamera::Ptr ncamera_loaded;

  std::string filename = "test_ncamera.yaml";
  YAML::Save(ncamera, filename);

  YAML::Load(filename, &ncamera_loaded);
  ASSERT_TRUE(ncamera_loaded.get() != nullptr);

  EXPECT_EQ(ncamera_loaded->getLabel(), ncamera->getLabel());
  EXPECT_EQ(ncamera_loaded->getId(), ncamera->getId());

  size_t num_cameras = ncamera_loaded->getNumCameras();
  EXPECT_EQ(num_cameras, 4u);

  for (size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
    const aslam::Camera& camera_loaded = ncamera_loaded->getCamera(cam_idx);
    const aslam::Camera& camera_gt = ncamera->getCamera(cam_idx);

    EXPECT_EQ(camera_loaded.getId(), camera_gt.getId());
    EXPECT_EQ(camera_loaded.getLabel(), camera_gt.getLabel());
    EXPECT_EQ(camera_loaded.imageHeight(), camera_gt.imageHeight());
    EXPECT_EQ(camera_loaded.imageWidth(), camera_gt.imageWidth());
    EXPECT_TRUE(EIGEN_MATRIX_NEAR(ncamera->get_T_C_B(cam_idx).getTransformationMatrix(),
                                  ncamera_loaded->get_T_C_B(cam_idx).getTransformationMatrix(),
                                  1e-8));
  }
}

ASLAM_UNITTEST_ENTRYPOINT
