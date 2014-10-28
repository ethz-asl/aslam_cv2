#include <gtest/gtest.h>

#include <aslam/cameras/ncamera-factory.h>

#include <aslam/common/entrypoint.h>

#include <memory>

using namespace aslam;

TEST(NCameraFactoryTest, testNCameraFactory) {
  NCamera::Ptr camera_rig = createPlanar4CameraRig();
  ASSERT_TRUE(static_cast<bool>(camera_rig));
  ASSERT_EQ(camera_rig->numCameras(), 4);
}

ASLAM_UNITTEST_ENTRYPOINT
