#include <iostream>
#include <aslam/cameras/distortion.h>
#include <glog/logging.h>

// TODO(slynen) Enable commented out PropertyTree support
//#include <sm/PropertyTree.hpp>

namespace aslam {
Distortion::Distortion(const Eigen::VectorXd& dist_coeffs)
    : distortion_coefficients_(dist_coeffs) {}

// TODO(slynen) Enable commented out PropertyTree support
//Distortion::Distortion(const sm::PropertyTree& /*property_tree*/) { }

bool Distortion::operator==(const Distortion& rhs) const {
  //check for same distortion type
  if (typeid(*this) != typeid(rhs))
    return false;

  if (distortion_coefficients_ != rhs.distortion_coefficients_)
    return false;

  return true;
}

void Distortion::distort(Eigen::Vector2d* point) const {
  CHECK_NOTNULL(point);
  distortUsingExternalCoefficients(distortion_coefficients_, point, nullptr);
}

void Distortion::distort(const Eigen::Vector2d& point,
                         Eigen::Vector2d* out_point) const {
  CHECK_NOTNULL(out_point);
  *out_point = point;
  distortUsingExternalCoefficients(distortion_coefficients_, out_point, nullptr);
}

void Distortion::distort(
    Eigen::Vector2d* point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const {
  CHECK_NOTNULL(point);
  CHECK_NOTNULL(out_jacobian);
  distortUsingExternalCoefficients(distortion_coefficients_, point, out_jacobian);
}

void Distortion::undistort(Eigen::Vector2d* point) const {
  CHECK_NOTNULL(point);
  undistortUsingExternalCoefficients(distortion_coefficients_, point);
}

void Distortion::undistort(const Eigen::Vector2d& point,
                           Eigen::Vector2d* out_point) const {
  CHECK_NOTNULL(out_point);
  *out_point = point;
  undistortUsingExternalCoefficients(distortion_coefficients_, out_point);
}

void Distortion::setParameters(const Eigen::VectorXd& dist_coeffs) {
  CHECK(distortionParametersValid(dist_coeffs)) << "Distortion parameters invalid!";
  distortion_coefficients_ = dist_coeffs;
}

Eigen::VectorXd Distortion::getParameters() const {
  return distortion_coefficients_;
}

}  // namespace aslam
