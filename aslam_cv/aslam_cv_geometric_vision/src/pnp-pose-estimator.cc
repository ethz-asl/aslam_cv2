#include <memory>

#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/camera-unified-projection.h>
#include <aslam/common/memory.h>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "aslam/geometric-vision/pnp-pose-estimator.h"

namespace aslam {
namespace geometric_vision {

bool PnpPoseEstimator::absolutePoseRansacPinholeCam(
    const Eigen::Matrix2Xd& measurements,
    const Eigen::Matrix3Xd& G_landmark_positions, double pixel_sigma,
    int max_ransac_iters, aslam::Camera::ConstPtr camera_ptr,
    aslam::Transformation* T_G_C, std::vector<int>* inliers, int* num_iters) {
  CHECK_NOTNULL(T_G_C);
  CHECK_NOTNULL(inliers);
  CHECK_NOTNULL(num_iters);
  CHECK_EQ(measurements.cols(), G_landmark_positions.cols());

  double ransac_threshold = 1;
  using aslam::PinholeCamera;
  using aslam::UnifiedProjectionCamera;
  switch (camera_ptr->getType()) {
    case aslam::Camera::Type::kPinhole: {
      const double fu =
          camera_ptr->getParameters()(PinholeCamera::Parameters::kFu);
      const double fv =
          camera_ptr->getParameters()(PinholeCamera::Parameters::kFv);

      const double focal_length = (fu + fv) / 2.0;
      ransac_threshold = 1.0 - cos(atan(pixel_sigma / focal_length));
      break;
    }
    case aslam::Camera::Type::kUnifiedProjection: {
      const double fu =
          camera_ptr->getParameters()(UnifiedProjectionCamera::Parameters::kFu);
      const double fv =
          camera_ptr->getParameters()(UnifiedProjectionCamera::Parameters::kFv);

      const double focal_length = (fu + fv) / 2.0;
      ransac_threshold = 1.0 - cos(atan(pixel_sigma / focal_length));
      break;
    }
    default:
      LOG(FATAL) << "Unknown camera type. The given camera is neither of type "
          "Pinhole nor UnifiedProjection.";
  }

  // Assuming the mean of lens focal lengths is the best estimate here.
  return absolutePoseRansac(measurements, G_landmark_positions,
                            ransac_threshold, max_ransac_iters, camera_ptr,
                            T_G_C, inliers, num_iters);
}

bool PnpPoseEstimator::absoluteMultiPoseRansacPinholeCam(
    const Eigen::Matrix2Xd& measurements,
    const std::vector<int>& measurement_camera_indices,
    const Eigen::Matrix3Xd& G_landmark_positions, double pixel_sigma,
    int max_ransac_iters, aslam::NCamera::ConstPtr ncamera_ptr,
    aslam::Transformation* T_G_I, std::vector<int>* inliers, int* num_iters) {
  std::vector<double> inlier_distances_to_model;
  return absoluteMultiPoseRansacPinholeCam(
      measurements, measurement_camera_indices, G_landmark_positions, pixel_sigma, max_ransac_iters,
      ncamera_ptr, T_G_I, inliers, &inlier_distances_to_model, num_iters);
}

bool PnpPoseEstimator::absoluteMultiPoseRansacPinholeCam(
    const Eigen::Matrix2Xd& measurements,
    const std::vector<int>& measurement_camera_indices,
    const Eigen::Matrix3Xd& G_landmark_positions, double pixel_sigma,
    int max_ransac_iters, aslam::NCamera::ConstPtr ncamera_ptr,
    aslam::Transformation* T_G_I, std::vector<int>* inliers,
    std::vector<double>* inlier_distances_to_model, int* num_iters) {
  CHECK_NOTNULL(T_G_I);
  CHECK_NOTNULL(inliers);
  CHECK_NOTNULL(inlier_distances_to_model);
  CHECK_NOTNULL(num_iters);
  CHECK_EQ(measurements.cols(), G_landmark_positions.cols());
  CHECK_EQ(measurements.cols(), static_cast<int>(measurement_camera_indices.size()));

  const size_t num_cameras = ncamera_ptr->getNumCameras();
  double focal_length = 0;

  // Average focal length together over all cameras in both axes.
  using aslam::PinholeCamera;
  using aslam::UnifiedProjectionCamera;
  for (size_t camera_index = 0; camera_index < num_cameras; ++camera_index) {
    const aslam::Camera::ConstPtr& camera_ptr =
        ncamera_ptr->getCameraShared(camera_index);
    switch (camera_ptr->getType()) {
      case aslam::Camera::Type::kPinhole: {
        const double fu =
            camera_ptr->getParameters()(PinholeCamera::Parameters::kFu);
        const double fv =
            camera_ptr->getParameters()(PinholeCamera::Parameters::kFv);
        focal_length += (fu + fv);
        break;
      }
      case aslam::Camera::Type::kUnifiedProjection: {
        const double fu =
            camera_ptr->getParameters()(UnifiedProjectionCamera::Parameters::kFu);
        const double fv =
            camera_ptr->getParameters()(UnifiedProjectionCamera::Parameters::kFv);

        focal_length += (fu + fv);
        break;
      }
      default:
        LOG(FATAL) << "Unknown camera type.  The given camera is neither of "
            "type Pinhole nor UnifiedProjection.";
    }
  }

  focal_length /= (2.0 * static_cast<double>(num_cameras));

  const double ransac_threshold = 1.0 - cos(atan(pixel_sigma / focal_length));

  return absoluteMultiPoseRansac(measurements, measurement_camera_indices,
                                 G_landmark_positions, ransac_threshold,
                                 max_ransac_iters, ncamera_ptr, T_G_I, inliers,
                                 inlier_distances_to_model, num_iters);
}

bool PnpPoseEstimator::absolutePoseRansac(
    const Eigen::Matrix2Xd& measurements,
    const Eigen::Matrix3Xd& G_landmark_positions, double ransac_threshold,
    int max_ransac_iters, aslam::Camera::ConstPtr camera_ptr,
    aslam::Transformation* T_G_C, std::vector<int>* inliers, int* num_iters) {
  CHECK_NOTNULL(T_G_C);
  CHECK_NOTNULL(inliers);
  CHECK_NOTNULL(num_iters);
  CHECK_EQ(measurements.cols(), G_landmark_positions.cols());

  opengv::points_t points;
  opengv::bearingVectors_t bearing_vectors;
  points.resize(measurements.cols());
  bearing_vectors.resize(measurements.cols());
  for (int i = 0; i < measurements.cols(); ++i) {
    camera_ptr->backProject3(measurements.col(i), &bearing_vectors[i]);
    bearing_vectors[i].normalize();
    points[i] = G_landmark_positions.col(i);
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors,
                                                        points);
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP,
              random_seed_));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = ransac_threshold;
  ransac.max_iterations_ = max_ransac_iters;
  bool ransac_success = ransac.computeModel();

  if (ransac_success) {
    T_G_C->getPosition() = ransac.model_coefficients_.rightCols(1);
    Eigen::Matrix<double, 3, 3> R_G_C(ransac.model_coefficients_.leftCols(3));
    T_G_C->getRotation() = aslam::Quaternion(R_G_C);
  }

  *inliers = ransac.inliers_;
  *num_iters = ransac.iterations_;
  return ransac_success;
}

bool PnpPoseEstimator::absoluteMultiPoseRansac(
    const Eigen::Matrix2Xd& measurements,
    const std::vector<int>& measurement_camera_indices,
    const Eigen::Matrix3Xd& G_landmark_positions, double ransac_threshold,
    int max_ransac_iters, aslam::NCamera::ConstPtr ncamera_ptr,
    aslam::Transformation* T_G_I, std::vector<int>* inliers, int* num_iters) {
  std::vector<double> inlier_distances_to_model;
  return absoluteMultiPoseRansac(
      measurements, measurement_camera_indices, G_landmark_positions, ransac_threshold,
      max_ransac_iters, ncamera_ptr, T_G_I, inliers, &inlier_distances_to_model, num_iters);
}

bool PnpPoseEstimator::absoluteMultiPoseRansac(
    const Eigen::Matrix2Xd& measurements,
    const std::vector<int>& measurement_camera_indices,
    const Eigen::Matrix3Xd& G_landmark_positions, double ransac_threshold,
    int max_ransac_iters, aslam::NCamera::ConstPtr ncamera_ptr,
    aslam::Transformation* T_G_I, std::vector<int>* inliers,
    std::vector<double>* inlier_distances_to_model, int* num_iters) {
  CHECK_NOTNULL(T_G_I);
  CHECK_NOTNULL(inliers);
  CHECK_NOTNULL(inlier_distances_to_model);
  CHECK_NOTNULL(num_iters);
  CHECK_EQ(measurements.cols(), G_landmark_positions.cols());
  CHECK_EQ(measurements.cols(), static_cast<int>(measurement_camera_indices.size()));

  // Fill in camera information from NCamera.
  // Rotation matrix for each camera.
  opengv::rotations_t cam_rotations;
  opengv::translations_t cam_translations;

  const int num_cameras = ncamera_ptr->getNumCameras();

  cam_rotations.resize(num_cameras);
  cam_translations.resize(num_cameras);

  for (int camera_index = 0; camera_index < num_cameras; ++camera_index) {
    const aslam::Transformation& T_C_B = ncamera_ptr->get_T_C_B(camera_index);
    // OpenGV requires body frame -> camera transformation.
    aslam::Transformation T_B_C = T_C_B.inverse();
    cam_rotations[camera_index] = T_B_C.getRotationMatrix();
    cam_translations[camera_index] = T_B_C.getPosition();
  }

  opengv::points_t points;
  opengv::bearingVectors_t bearing_vectors;
  points.resize(measurements.cols());
  bearing_vectors.resize(measurements.cols());
  for (int i = 0; i < measurements.cols(); ++i) {
    // Figure out which camera this corresponds to, and reproject it in the
    // correct camera.
    int camera_index = measurement_camera_indices[i];
    ncamera_ptr->getCamera(camera_index)
        .backProject3(measurements.col(i), &bearing_vectors[i]);
    bearing_vectors[i].normalize();
    points[i] = G_landmark_positions.col(i);
  }
  // Basically same as the Central, except measurement_camera_indices, which
  // assigns a camera index to each bearing_vector, and cam_offsets and
  // cam_rotations, which describe the position and orientation of the cameras
  // with respect to the body frame.
  opengv::absolute_pose::NoncentralAbsoluteAdapter adapter(
      bearing_vectors, measurement_camera_indices, points, cam_translations,
      cam_rotations);
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(adapter,
              opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::GP3P,
              random_seed_));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = ransac_threshold;
  ransac.max_iterations_ = max_ransac_iters;
  bool ransac_success = ransac.computeModel();
  CHECK_EQ(ransac.inliers_.size(), ransac.inlier_distances_to_model_.size());

  if (ransac_success) {
    // Optional nonlinear model refinement over all inliers.
    Eigen::Matrix<double, 3, 4> final_model = ransac.model_coefficients_;
    if (run_nonlinear_refinement_) {
      absposeproblem_ptr->optimizeModelCoefficients(ransac.inliers_,
                                                    ransac.model_coefficients_,
                                                    final_model);
    }

    // Set result.
    T_G_I->getPosition() = final_model.rightCols(1);
    Eigen::Matrix<double, 3, 3> R_G_I(final_model.leftCols(3));
    T_G_I->getRotation() = aslam::Quaternion(R_G_I);
  }

  *inliers = ransac.inliers_;
  *inlier_distances_to_model = ransac.inlier_distances_to_model_;
  *num_iters = ransac.iterations_;

  return ransac_success;
}

}  // namespace geometric_vision
}  // namespace aslam
