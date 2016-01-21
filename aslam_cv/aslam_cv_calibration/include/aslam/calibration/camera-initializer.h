#ifndef ASLAM_CALIBRATION_CAMERA_INITIALIZER_H
#define ASLAM_CALIBRATION_CAMERA_INITIALIZER_H

#include <glog/logging.h>

#include <aslam/calibration/camera-initializer-pinhole-helpers.h>
#include <aslam/calibration/target-observation.h>
#include <aslam/cameras/camera-factory.h>
#include <aslam/common/stl-helpers.h>

namespace aslam {
namespace calibration {

class CameraInitializer  {
 public:
  /// General function to catch not specialized camera types.
  template<typename CameraType>
  bool initializeCameraFromTargetObservation(
    TargetObservation obs,
    std::shared_ptr<CameraType>* camera) {
   CHECK_NOTNULL(camera);
   LOG(FATAL) << "Initialization not implemented for this camera type";
   return false;
  }

  /// Function to initialize Intrinsics vector. TODO: work on code
  /// \brief initialize the intrinsics based on one view of a gridded calibration target
  /// \return true on success
  /// These functions are based on functions from Lionel Heng and the excellent camodocal
  /// https://github.com/hengli/camodocal
  //this algorithm can be used with high distortion lenses
  bool initializeCameraFromTargetObservation<aslam::PinholeCamera>(
    TargetObservation obs,
    std::shared_ptr<aslam::PinholeCamera>* camera) {
   CHECK_NOTNULL(camera) << "No camera specified";
   CHECK_NOTNULL(obs.target_) << "No target found";

   // First, initialize the image center at the center of the image.
   camera->get()->intrinsics_[2] = (obs.getTarget()->cols() - 1.0) / 2.0;
   camera->get()->intrinsics_[3] = (obs.getTarget()->rows() - 1.0) / 2.0;
   //_ru = obs.target_->cols();
   //_rv = obs.target_->rows();
   camera->get()->distortion_.reset(new NullDistortion);

   // Initialize focal length
   // C. Hughes, P. Denny, M. Glavin, and E. Jones,
   // Equidistant Fish-Eye Calibration and Rectification by Vanishing Point
   // Extraction, PAMI 2010
   // Find circles from rows of chessboard corners, and for each pair
   // of circles, find vanishing points: v1 and v2.
   // f = ||v1 - v2|| / PI;
   std::vector<double> f_guesses;

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> center(obs.getTarget()->rows());
  double radius[obs.getTarget()->rows()];
  bool skipImage=false;

  for (size_t r=0; r<obs.getTarget()->rows(); ++r) {
    std::vector<cv::Point2d> circle;
    for (size_t c=0; c<obs.getTarget()->cols(); ++c) {
      Eigen::Vector2d imagePoint;
      Eigen::Vector3d gridPoint;

      if (obs.imageGridPoint(r, c, imagePoint))
        circle.emplace_back(cv::Point2f(imagePoint[0], imagePoint[1]));
      else
        //skip this image if the board view is not complete
        skipImage=true;
    }
    PinholeHelpers::fitCircle(circle, center[r](0), center[r](1), radius[r]);


    if(skipImage)
      continue;

    for (size_t j=0; j<obs.getTarget()->rows(); ++j) {
      for (size_t k=j+1; k<obs.getTarget()->cols(); ++k) {
        // find distance between pair of vanishing points which
        // correspond to intersection points of 2 circles
        std::vector < cv::Point2d > ipts;
        ipts = PinholeHelpers::intersectCircles(center[j](0), center[j](1),
                                                radius[j], center[k](0), center[k](1), radius[k]);
        if (ipts.size()<2)
          continue;

        double f_guess = cv::norm(ipts.at(0) - ipts.at(1)) / M_PI;
        f_guesses.emplace_back(f_guess);
      }
    }
  }

  //get the median of the guesses
  if(f_guesses.empty())
    return false;
  double f0 = aslam::common::median(f_guesses.begin(), f_guesses.end());

  //set the estimate
  camera->get()->intrinsics_[0] = f0;
  camera->get()->intrinsics_[1] = f0;

  return true;
  }

 private:

};

}  // namespace calibration
}  // namespace aslam


#endif  // ASLAM_CALIBRATION_CAMERA_INITIALIZER_H
