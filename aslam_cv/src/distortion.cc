#include <iostream>
#include <aslam/cameras/distortion.h>
#include <glog/logging.h>

// TODO(slynen) Enable commented out PropertyTree support
//#include <sm/PropertyTree.hpp>

namespace aslam {
Distortion::Distortion(const Eigen::VectorXd& dist_coeffs)
    : dist_coeffs_(dist_coeffs) {
}

// TODO(slynen) Enable commented out PropertyTree support
//Distortion::Distortion(const sm::PropertyTree& /*property_tree*/) { }
Distortion::~Distortion() {
}

bool Distortion::operator==(const Distortion& rhs) const {
  //check for same distortion type
  if(typeid(*this) != typeid(rhs))
    return false;

  //check for same distortion parameters
  if(dist_coeffs_ != rhs.dist_coeffs_)
    return false;

  return true;
}

void Distortion::distort(Eigen::Vector2d* point) const {
  CHECK_NOTNULL(point);
  distortExternalCoeffs(dist_coeffs_, point, NULL);
}

void Distortion::distort(const Eigen::Vector2d& point,
                         Eigen::Vector2d* out_point) const {
  CHECK_NOTNULL(out_point);
  *out_point = point;
  distortExternalCoeffs(dist_coeffs_, out_point, NULL);
}

void Distortion::distort(
    Eigen::Vector2d* point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const {
  CHECK_NOTNULL(point);
  CHECK_NOTNULL(out_jacobian);
  distortExternalCoeffs(dist_coeffs_, point, out_jacobian);
}

void Distortion::undistort(Eigen::Vector2d* point) const {
  CHECK_NOTNULL(point);
  undistortExternalCoeffs(dist_coeffs_, point, NULL);
}

void Distortion::undistort(const Eigen::Vector2d& point,
                           Eigen::Vector2d* out_point) const {
  CHECK_NOTNULL(out_point);
  *out_point = point;
  undistortExternalCoeffs(dist_coeffs_, out_point, NULL);
}

void Distortion::undistort(
    Eigen::Vector2d* point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const {
  CHECK_NOTNULL(point);
  CHECK_NOTNULL(out_jacobian);
  distortExternalCoeffs(dist_coeffs_, point, out_jacobian);
}

void Distortion::setParameters(const Eigen::VectorXd& params) {
  CHECK(distortionParametersValid(params)) << "Distortion parameters invalid!";
  dist_coeffs_ = params;
}

Eigen::VectorXd Distortion::getParameters() const {
  return dist_coeffs_;
}

}  // namespace aslam
