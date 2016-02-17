#ifndef ASLAM_CALIBRATION_CAMERA_INITIALIZER_H
#define ASLAM_CALIBRATION_CAMERA_INITIALIZER_H

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <opencv2/calib3d/calib3d.hpp>

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

///// Initializes the intrinsics vector based on one view of a gridded calibration target.
///// On success it returns true.
///// These functions are based on functions from Lionel Heng and the excellent camodocal
///// https://github.com/hengli/camodocal.
///// This algorithm can be used with high distortion lenses.
//template<>
//bool initializeCameraIntrinsics<aslam::PinholeCamera>(
//    Eigen::VectorXd& intrinsics_vector,
//    std::vector<TargetObservation::Ptr> &observations) {
//  CHECK_NOTNULL(&intrinsics_vector);
//  CHECK(!observations.size() == 0) << "Need min. one observation.";
//
//  //process all images
//  size_t nImages = observations.size();
//
//  // Initialize focal length
//  // C. Hughes, P. Denny, M. Glavin, and E. Jones,
//  // Equidistant Fish-Eye Calibration and Rectification by Vanishing Point
//  // Extraction, PAMI 2010
//  // Find circles from rows of chessboard corners, and for each pair
//  // of circles, find vanishing points: v1 and v2.
//  // f = ||v1 - v2|| / PI;
//  std::vector<double> f_guesses;
//
//  for (size_t i=0; i<nImages; ++i) {
//    TargetObservation::Ptr obs = observations.at(i);
//    CHECK(!obs->getTarget()->size() == 0) << "The TargetObservation has no target object.";
//    const TargetBase current_target = *obs->getTarget();
//
//    aslam::Aligned<std::vector, Eigen::Vector2d>::type center(current_target.rows());
//    double * radius = new double[current_target.rows()];
//    bool skip_image = false;
//
//   std::vector<cv::Point2f> image_corners(obs->numObservedCorners());
//   for (size_t j = 0; j < image_corners.size(); ++j) {
//     image_corners.at(j) = cv::Point2f(obs->getObservedCorner(j)[0], obs->getObservedCorner(j)[1]);
//   }
//    for (size_t r = 0; r < current_target.rows(); ++r) {
//      std::vector<cv::Point2d> circle;
//      for (size_t c = 0; c < current_target.cols(); ++c) {
//        if (obs->checkIdinImage(r, c)) {
//          circle.push_back(image_corners.at(r * current_target.cols() + c));
//        }
//        else {
//          // Skips this image if corner id not part of image id set.
//          skip_image = true;
//        }
//      }
//      PinholeHelpers::fitCircle(circle, center[r](0), center[r](1), radius[r]);
//    }
//
//    if(skip_image){
//      delete [] radius;
//      continue;
//    }
//
//    for (size_t j = 0; j < current_target.rows(); ++j) {
//      for (size_t k = j + 1; k < current_target.cols(); ++k) {
//        // Find the distance between pair of vanishing points which
//        // correspond to intersection points of 2 circles.
//        std::vector<cv::Point2d> ipts;
//        PinholeHelpers::intersectCircles(ipts,
//                                         center[j](0), center[j](1), radius[j],
//                                         center[k](0), center[k](1), radius[k]);
//        if (ipts.size() < 2) {
//          continue;
//        }
//        double f_guess = cv::norm(ipts.at(0) - ipts.at(1)) / M_PI;
//        f_guesses.emplace_back(f_guess);
//      }
//    }
//
//    // Frees allocated memory.
//    delete [] radius;
//  }
//
//  // Gets the median of the guesses.
//  if(f_guesses.empty()) { return false; }
//  double f0 = aslam::common::median(f_guesses.begin(), f_guesses.end());
//  double f0_mean = aslam::common::mean(f_guesses.begin(), f_guesses.end());
//  double f0_std = aslam::common::stdev(f_guesses.begin(), f_guesses.end());
//  std::cout << "mean =  " << f0_mean << " / standard deviation =  " << f0_std << "\n\n";
//
//  // Sets the first intrinsics estimate.
//  intrinsics_vector.resize(aslam::PinholeCamera::parameterCount());
//  intrinsics_vector(PinholeCamera::kFu) = f0;
//  intrinsics_vector(PinholeCamera::kFv) = f0;
//  intrinsics_vector(PinholeCamera::kCu) = (observations.at(0)->getImageWidth() - 1.0) / 2.0;
//  intrinsics_vector(PinholeCamera::kCv) = (observations.at(0)->getImageHeight() - 1.0) / 2.0;
//
//  return true;
//}


/// Initializes the intrinsics vector based on one view of a gridded calibration target.
/// On success it returns true.
/// These functions are based on functions from Lionel Heng and the excellent camodocal
/// https://github.com/hengli/camodocal.
template<>
bool initializeCameraIntrinsics<aslam::PinholeCamera>(
    Eigen::VectorXd& intrinsics_vector,
    std::vector<TargetObservation::Ptr> &observations) {
 CHECK_NOTNULL(&intrinsics_vector);
 CHECK(!observations.size() == 0) << "Need min. one observation.";

 //image centers
 double cu = (observations.at(0)->getImageWidth() - 1.0) / 2.0;
 double cv = (observations.at(0)->getImageHeight() - 1.0) / 2.0;

 //process all images
 size_t nImages = observations.size();

 // Initialize intrinsics
 // Z. Zhang
 // A Flexible New Technique for Camera Calibration,
 // Extraction, PAMI 2000
 // Intrinsics estimation with image of absolute conic

 cv::Mat A(nImages * 2, 2, CV_64F);
 cv::Mat b(nImages * 2, 1, CV_64F);

 for (size_t i=0; i<nImages; ++i) {
   TargetObservation::Ptr obs = observations.at(i);
   CHECK(!obs->getTarget()->size() == 0) << "The TargetObservation has no target object.";
   const TargetBase current_target = *obs->getTarget();

   std::vector<cv::Point2f> image_corners(obs->numObservedCorners());
   for (size_t j = 0; j < image_corners.size(); ++j) {
     image_corners.at(j) = cv::Point2f(obs->getObservedCorner(j)[0], obs->getObservedCorner(j)[1]);
   }

   std::vector<cv::Point2f> M(current_target.size());
   for (size_t j = 0; j < M.size(); ++j) {
     M.at(j) = cv::Point2f(current_target.point(j)[0], current_target.point(j)[1]);
   }

   cv::Mat H = cv::findHomography(M, image_corners);

   H.at<double>(0,0) -= H.at<double>(2,0) * cu;
   H.at<double>(0,1) -= H.at<double>(2,1) * cu;
   H.at<double>(0,2) -= H.at<double>(2,2) * cu;
   H.at<double>(1,0) -= H.at<double>(2,0) * cv;
   H.at<double>(1,1) -= H.at<double>(2,1) * cv;
   H.at<double>(1,2) -= H.at<double>(2,2) * cv;

   double h[3], v[3], d1[3], d2[3];
   double n[4] = {0,0,0,0};

   for (int j = 0; j < 3; ++j) {
     double t0 = H.at<double>(j,0);
     double t1 = H.at<double>(j,1);
     h[j] = t0; v[j] = t1;
     d1[j] = (t0 + t1) * 0.5;
     d2[j] = (t0 - t1) * 0.5;
     n[0] += t0 * t0; n[1] += t1 * t1;
     n[2] += d1[j] * d1[j]; n[3] += d2[j] * d2[j];
   }

   for (int j = 0; j < 4; ++j) {
     n[j] = 1.0 / sqrt(n[j]);
   }

   for (int j = 0; j < 3; ++j) {
     h[j] *= n[0]; v[j] *= n[1];
     d1[j] *= n[2]; d2[j] *= n[3];
   }

   A.at<double>(i * 2, 0) = h[0] * v[0];
   A.at<double>(i * 2, 1) = h[1] * v[1];
   A.at<double>(i * 2 + 1, 0) = d1[0] * d2[0];
   A.at<double>(i * 2 + 1, 1) = d1[1] * d2[1];
   b.at<double>(i * 2, 0) = -h[2] * v[2];
   b.at<double>(i * 2 + 1, 0) = -d1[2] * d2[2];
 }

 cv::Mat f(2, 1, CV_64F);
 cv::solve(A, b, f, cv::DECOMP_NORMAL | cv::DECOMP_LU);

 // Sets the first intrinsics estimate.
 intrinsics_vector.resize(aslam::PinholeCamera::parameterCount());
 intrinsics_vector(PinholeCamera::kFu) = sqrt(fabs(1.0 / f.at<double>(0)));
 intrinsics_vector(PinholeCamera::kFv) = sqrt(fabs(1.0 / f.at<double>(1)));
 intrinsics_vector(PinholeCamera::kCu) = (observations.at(0)->getImageWidth() - 1.0) / 2.0;
 intrinsics_vector(PinholeCamera::kCv) = (observations.at(0)->getImageHeight() - 1.0) / 2.0;

 return true;
}

}  // namespace calibration
}  // namespace aslam


#endif  // ASLAM_CALIBRATION_CAMERA_INITIALIZER_H
