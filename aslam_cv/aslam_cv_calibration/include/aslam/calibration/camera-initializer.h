#ifndef ASLAM_CALIBRATION_CAMERA_INITIALIZER_H
#define ASLAM_CALIBRATION_CAMERA_INITIALIZER_H

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <glog/logging.h>

#include <aslam/calibration/camera-initializer-pinhole-helpers.h>
#include <aslam/calibration/target-observation.h>
#include <aslam/cameras/camera-factory.h>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/common/memory.h>
#include <aslam/common/stl-helpers.h>

namespace aslam {
namespace calibration {

template<typename CameraType>
bool initializeCameraIntrinsics(
    Eigen::VectorXd& intrinsics_vector,
    std::vector<TargetObservation::Ptr> &observations) {
 LOG(FATAL) << "Initialization not implemented for this camera type";
 return false;
}

/// Initializes the intrinsics vector based on one view of a gridded calibration target.
/// On success it returns true.
/// These functions are based on functions from Lionel Heng and the excellent camodocal
/// https://github.com/hengli/camodocal.
/// This algorithm can be used with high distortion lenses.
template<>
bool initializeCameraIntrinsics<aslam::PinholeCamera>(
    Eigen::VectorXd& intrinsics_vector,
    std::vector<TargetObservation::Ptr> &observations) {
 CHECK_NOTNULL(&intrinsics_vector);
 CHECK(!observations.size() == 0) << "Need min. one observation.";

  //process all images
  size_t nImages = observations.size();

  // Initialize focal length
  // C. Hughes, P. Denny, M. Glavin, and E. Jones,
  // Equidistant Fish-Eye Calibration and Rectification by Vanishing Point
  // Extraction, PAMI 2010
  // Find circles from rows of chessboard corners, and for each pair
  // of circles, find vanishing points: v1 and v2.
  // f = ||v1 - v2|| / PI;
  std::vector<double> f_guesses;

  for (size_t i=0; i<nImages; ++i) {
    TargetObservation::Ptr obs = observations.at(i);
    CHECK(!obs->getTarget()->size() == 0) << "The TargetObservation has no target object.";
    const TargetBase current_target = *obs->getTarget();

    aslam::Aligned<std::vector, Eigen::Vector2d>::type center(current_target.rows());
    double * radius = new double[current_target.rows()];
    bool skip_image = false;

    for (size_t r = 0; r < current_target.rows(); ++r) {
      std::vector<cv::Point2d> circle;
      for (size_t c = 0; c < current_target.cols(); ++c) {
        Eigen::Vector2d image_point;
        Eigen::Vector3d grid_point;

        if (obs->checkIdinImage(r, c, image_point)) {
          circle.emplace_back(cv::Point2d(image_point(0), image_point(1)));
        }
        else {
          // Skips this image if corner id not part of image id set.
          skip_image = true;
        }
      }
      PinholeHelpers::fitCircle(circle, center[r](0), center[r](1), radius[r]);
    }

    if(skip_image){
      delete [] radius;
      continue;
    }

   for (size_t j = 0; j < current_target.rows(); ++j) {
     for (size_t k = j + 1; k < current_target.cols(); ++k) {
       // Find the distance between pair of vanishing points which
       // correspond to intersection points of 2 circles.
       std::vector<cv::Point2d>* ipts;
       ipts = PinholeHelpers::intersectCircles(center[j](0), center[j](1),
                                               radius[j], center[k](0),
                                               center[k](1), radius[k]);
       if (ipts->size() < 2) { continue; }

       double f_guess = cv::norm(ipts->at(0) - ipts->at(1)) / M_PI;
       f_guesses.emplace_back(f_guess);
     }
   }

   // Frees allocated memory.
   delete [] radius;
 }

   // Gets the median of the guesses.
   if(f_guesses.empty()) { return false; }
   double f0 = aslam::common::median(f_guesses.begin(), f_guesses.end());

   // Sets the first intrinsics estimate.
   intrinsics_vector.resize(aslam::PinholeCamera::parameterCount());
   intrinsics_vector(PinholeCamera::kFu) = f0;
   intrinsics_vector(PinholeCamera::kFv) = f0;
   intrinsics_vector(PinholeCamera::kCu) = (observations.at(0)->getImageWidth() - 1.0) / 2.0;
   intrinsics_vector(PinholeCamera::kCv) = (observations.at(0)->getImageHeight() - 1.0) / 2.0;

   return true;
  }

}  // namespace calibration
}  // namespace aslam


#endif  // ASLAM_CALIBRATION_CAMERA_INITIALIZER_H
