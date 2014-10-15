#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>

#include <aslam/cameras/camera.h>
#include <aslam/common/pose-types.h>
#include <aslam/geometric-vision/five-point-pose-estimator.h>

namespace aslam {
namespace geometric_vision {

int FivePointPoseEstimator::computePinhole(const Eigen::Matrix2Xd& measurements_a,
                                           const Eigen::Matrix2Xd& measurements_b,
                                           std::shared_ptr<Camera> camera_ptr, double pixel_sigma,
                                           unsigned int max_ransac_iters,
                                           aslam::Transformation* output_transform) {
  CHECK(camera_ptr);
  CHECK_NOTNULL(output_transform);
  CHECK_EQ(measurements_a.cols(), measurements_b.cols());

  // This method is designed for pinhole camera, so we should be able to cast
  // the base camera ptr to the derived pinhole type.
  aslam::PinholeCamera::Ptr pinhole_camera_ptr = std::dynamic_pointer_cast<aslam::PinholeCamera>(
      camera_ptr);
  CHECK(pinhole_camera_ptr) << "Couldn't cast camera pointer to pinhole camera type.";

  // Assuming the mean of lens focal lengths is the best estimate here.
  const double focal_length = (pinhole_camera_ptr->fu() + pinhole_camera_ptr->fv()) / 2.0;
  const double ransac_threshold = 1.0 - cos(atan(pixel_sigma / focal_length));

  return FivePointPoseEstimator::compute(measurements_a, measurements_b, camera_ptr, camera_ptr,
                                         ransac_threshold, max_ransac_iters, output_transform);
}

int FivePointPoseEstimator::compute(const Eigen::Matrix2Xd& measurements_a,
                                    const Eigen::Matrix2Xd& measurements_b,
                                    std::shared_ptr<Camera> camera_A,
                                    std::shared_ptr<Camera> camera_B, double ransac_threshold,
                                    unsigned int max_ransac_iters,
                                    aslam::Transformation* output_transform) {
  CHECK(camera_A);
  CHECK(camera_B);
  CHECK_NOTNULL(output_transform);
  CHECK_EQ(measurements_a.cols(), measurements_b.cols());
  CHECK_GE(measurements_a.cols(), 5);

  opengv::bearingVectors_t bearing_vectors_a;
  opengv::bearingVectors_t bearing_vectors_b;
  bearing_vectors_a.resize(measurements_a.cols());
  bearing_vectors_b.resize(measurements_b.cols());

  for (unsigned int i = 0; i < measurements_a.cols(); ++i) {
    bool success = camera_A->backProject3(measurements_a.col(i), &bearing_vectors_a[i]);
    bearing_vectors_a[i].normalize();

    success &= camera_B->backProject3(measurements_b.col(i), &bearing_vectors_b[i]);
    bearing_vectors_b[i].normalize();

    CHECK(success) << "backprojection failed!";
  }

  opengv::rotation_t rotation;
  rotation.setIdentity();

  // create a central relative adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(bearing_vectors_a, bearing_vectors_b,
                                                        rotation);

  opengv::sac::Ransac<opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> ransac;
  boost::shared_ptr<opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> relposeproblem_ptr(
      new opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem(
          adapter, opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::NISTER));
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_threshold;
  ransac.max_iterations_ = max_ransac_iters;

  ransac.computeModel();

  int num_inliers = ransac.inliers_.size();

  output_transform->getPosition() = ransac.model_coefficients_.rightCols(1);
  const Eigen::Matrix3d A_R_B = ransac.model_coefficients_.leftCols(3);

  // Eigen constructs active quaternion when constructing it from
  // rotation matrix, but we want to keep passive quaternions everywhere.
  output_transform->getRotation().toImplementation() = Eigen::Quaterniond(A_R_B.transpose())
      .normalized();

  return num_inliers;
}

}  //namespace geometric_vision
}  //namespace aslam

