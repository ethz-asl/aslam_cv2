#ifndef ASLAM_CALIBRATION_CAMERA_INITIALIZER_H
#define ASLAM_CALIBRATION_CAMERA_INITIALIZER_H

#include <glog/logging.h>

#include <aslam/calibration/camera-initializer-pinhole-helpers.h>
#include <aslam/calibration/target-observation.h>
#include <aslam/cameras/camera-factory.h>
#include <aslam/common/memory.h>
#include <aslam/common/stl-helpers.h>

namespace aslam {
namespace calibration {

class CameraInitializer  {
 public:

  template<typename CameraType>
  bool initializeCameraFromTargetObservation(
    TargetObservation obs,
    std::shared_ptr<CameraType>* camera) {
   CHECK_NOTNULL(camera);
   LOG(FATAL) << "Initialization not implemented for this camera type";
   return false;
  }

  /// Initializes the intrinsics vector based on one view of a gridded calibration target.
  /// On success it returns true.
  /// These functions are based on functions from Lionel Heng and the excellent camodocal
  /// https://github.com/hengli/camodocal.
  /// This algorithm can be used with high distortion lenses.
  template<>
  bool initializeCameraFromTargetObservation<aslam::PinholeCamera>(
    TargetObservation obs,
    std::shared_ptr<aslam::PinholeCamera>* camera) {
   CHECK(!camera == nullptr) << "No camera specified";
   CHECK_NOTNULL(obs.getTarget()) << "No target found";

   // First, initialize the image center at the center of the image.
   camera->get()->intrinsics_[camera->get()->Parameters::kCu] = (obs.getTarget()->cols() - 1.0) / 2.0;
   camera->get()->intrinsics_[camera->get()->Parameters::kCv] = (obs.getTarget()->rows() - 1.0) / 2.0;
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

   aslam::Aligned center(obs.getTarget()->rows());
   double* radius = &new double[];
   bool skip_image = false;

   for (size_t r = 0; r < obs.getTarget()->rows(); ++r) {
     std::vector<cv::Point2d> circle;
     for (size_t c = 0; c < obs.getTarget()->cols(); ++c) {
       Eigen::Vector2d image_point;
       Eigen::Vector3d grid_point;

       if (obs.imageGridPoint(r, c, image_point))
         circle.emplace_back(cv::Point2f(image_point[0], image_point[1]));
       else
         //Skips this image if the board view is not complete.
         skip_image=true;
     }
     PinholeHelpers::fitCircle(circle, center[r](0), center[r](1), radius[r]);

     if(skip_image)
       continue;

     for (size_t j = 0; j < obs.getTarget()->rows(); ++j) {
       for (size_t k = j + 1; k < obs.getTarget()->cols(); ++k) {
         // Find the distance between pair of vanishing points which
         // correspond to intersection points of 2 circles.
         std::vector<cv::Point2d> ipts;
         ipts = PinholeHelpers::intersectCircles(center[j](0), center[j](1),
                                                 radius[j], center[k](0),
                                                 center[k](1), radius[k]);
         if (ipts.size() < 2) {
           continue;
         }
         double f_guess = cv::norm(ipts.at(0) - ipts.at(1)) / M_PI;
         f_guesses.emplace_back(f_guess);
       }
     }
   }

   //Frees allocated memory.
   delete [] radius;

   //Gets the median of the guesses.
   if(f_guesses.empty()) {
     return false;
   }
   double f0 = aslam::common::median(f_guesses.begin(), f_guesses.end());

   //Sets the estimate.
   camera->get()->intrinsics_[camera->get()->Parameters::kFu] = f0;
   camera->get()->intrinsics_[camera->get()->Parameters::kFv] = f0;

   return true;
  }
};

}  // namespace calibration
}  // namespace aslam


#endif  // ASLAM_CALIBRATION_CAMERA_INITIALIZER_H
