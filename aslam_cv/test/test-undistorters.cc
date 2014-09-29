#include <cmath>

#include <Eigen/Core>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <aslam/cameras/camera-unified-projection.h>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/common/eigen-helpers.h>
#include <aslam/common/eigen-predicates.h>
#include <aslam/common/entrypoint.h>
#include <aslam/pipeline/undistorter-mapped.h>
#include <aslam/common/memory.h>

///////////////////////////////////////////////
// Types to test
///////////////////////////////////////////////
using testing::Types;
typedef Types<aslam::PinholeCamera,
              aslam::UnifiedProjectionCamera> Implementations;

///////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////
template <class _CameraType>
class TestUndistorters : public testing::Test {
 public:
  typedef _CameraType CameraType;

  protected:
  TestUndistorters() : camera_(CameraType::template createTestCamera<aslam::RadTanDistortion>() ) {};
    virtual ~TestUndistorters() {};
    typename CameraType::Ptr camera_;
};

TYPED_TEST_CASE(TestUndistorters, Implementations);

///////////////////////////////////////////////
// Generic test cases (run for all models)
///////////////////////////////////////////////

TYPED_TEST(TestUndistorters, TestMappedUndistorter) {
  std::unique_ptr<aslam::MappedUndistorter> undistorter = this->camera_->createMappedUndistorter(
      1.0, 1.0, aslam::InterpolationMethod::Linear);

  // Get the undistortion maps.
  const cv::Mat& map_u = undistorter->getUndistortMapU();
  const cv::Mat& map_v = undistorter->getUndistortMapV();

  // Test the undistortion on some random points.
  const int num_points = 20;
  for(int i=0; i<num_points; i++)
  {
    // Convert map to non-fixed point representation for easy lookup of values.
    cv::Mat map_u_copy = map_u.clone();
    cv::Mat map_v_copy = map_v.clone();
    cv::convertMaps(map_u, map_v, map_u_copy, map_v_copy, CV_32FC1);

    // Create random keypoint (round the coordinates to avoid interpolation on the maps)
    Eigen::Vector2d keypoint_undistorted = this->camera_->createRandomKeypoint();
    keypoint_undistorted[0] = std::floor(keypoint_undistorted[0]);
    keypoint_undistorted[1] = std::floor(keypoint_undistorted[1]);

    // Distort using internal camera functions.
    Eigen::Vector2d keypoint_distorted;
    Eigen::Vector3d point_3d;
    undistorter->getOutputCamera().backProject3(keypoint_undistorted, &point_3d);
    point_3d /= point_3d[2];
    this->camera_->project3(point_3d, &keypoint_distorted);

    // Distort using the maps.
    auto query_map = [map_u_copy, map_v_copy](double u, float v) {
      const double u_map = map_u_copy.at<float>(v, u);
      const double v_map = map_v_copy.at<float>(v, u);
      return Eigen::Vector2d(u_map, v_map);
    };

    Eigen::Vector2d keypoint_distorted_maps = query_map(std::floor(keypoint_undistorted[0]),
                                                        std::floor(keypoint_undistorted[1]));

    // As we only use full integer lookup in the map and do not interpolate we use a good tolerance.
    EXPECT_NEAR_EIGEN(keypoint_distorted, keypoint_distorted_maps, 0.05);
  }
}

////////////////////////////////////
// Camera model specific test cases
////////////////////////////////////

TEST(TestUndistortersUpm, TestMappedUndistorterUdfToPinhole) {
  aslam::UnifiedProjectionCamera::Ptr camera = aslam::UnifiedProjectionCamera::createTestCamera<
      aslam::RadTanDistortion>();

  std::unique_ptr<aslam::MappedUndistorter> undistorter = camera->createMappedUndistorterToPinhole(
      1.0, 1.0, aslam::InterpolationMethod::Linear);

  // Get the undistortion maps.
  const cv::Mat& map_u = undistorter->getUndistortMapU();
  const cv::Mat& map_v = undistorter->getUndistortMapV();

  // Test the undistortion on some random points.
  const int num_points = 20;
  for(int i=0; i<num_points; i++)
  {
    // Convert map to non-fixed point representation for easy lookup of values.
    cv::Mat map_u_copy = map_u.clone();
    cv::Mat map_v_copy = map_v.clone();
    cv::convertMaps(map_u, map_v, map_u_copy, map_v_copy, CV_32FC1);

    // Create random keypoint (round the coordinates to avoid interpolation on the maps)
    Eigen::Vector2d keypoint_undistorted = camera->createRandomKeypoint();
    keypoint_undistorted[0] = std::floor(keypoint_undistorted[0]);
    keypoint_undistorted[1] = std::floor(keypoint_undistorted[1]);

    // Distort using internal camera functions.
    Eigen::Vector2d keypoint_distorted;
    Eigen::Vector3d point_3d;
    undistorter->getOutputCamera().backProject3(keypoint_undistorted, &point_3d);
    point_3d /= point_3d[2];
    camera->project3(point_3d, &keypoint_distorted);

    // Distort using the maps.
    auto query_map = [map_u_copy, map_v_copy](double u, float v) {
      const double u_map = map_u_copy.at<float>(v, u);
      const double v_map = map_v_copy.at<float>(v, u);
      return Eigen::Vector2d(u_map, v_map);
    };

    Eigen::Vector2d keypoint_distorted_maps = query_map(std::floor(keypoint_undistorted[0]),
                                                        std::floor(keypoint_undistorted[1]));

    // As we only use full integer lookup in the map and do not interpolate we use a good tolerance.
    EXPECT_NEAR_EIGEN(keypoint_distorted, keypoint_distorted_maps, 0.05);
  }
}

