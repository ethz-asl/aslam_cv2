#include <iostream>
#include <limits>
#include <aslam/cameras/equidistant-distortion.h>

namespace aslam {

EquidistantDistortion::EquidistantDistortion(const Eigen::VectorXd& dist_coeffs)
: Distortion(dist_coeffs) {
  CHECK(distortionParametersValid(dist_coeffs)) << "Invalid distortion parameters!";
}

void EquidistantDistortion::distortUsingExternalCoefficients(
    const Eigen::VectorXd& dist_coeffs,
    Eigen::Vector2d* point,
    Eigen::Matrix2d* out_jacobian) const {
  CHECK_EQ(dist_coeffs.size(), kNumOfParams) << "dist_coeffs: invalid size!";
  CHECK_NOTNULL(point);

  double& x = (*point)(0);
  double& y = (*point)(1);

  const double& k1 = dist_coeffs(0);
  const double& k2 = dist_coeffs(1);
  const double& k3 = dist_coeffs(2);
  const double& k4 = dist_coeffs(3);

  double x2 = x*x;
  double y2 = y*y;
  double r = sqrt(x2 + y2);
  double theta = atan(r);
  double theta2 = theta * theta;
  double theta4 = theta2 * theta2;
  double theta6 = theta2 * theta4;
  double theta8 = theta4 * theta4;
  double thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

  if(out_jacobian)
  {
    double theta3 = theta2 * theta;
    double theta5 = theta4 * theta;
    double theta7 = theta6 * theta;

    //MATLAB generated Jacobian
    const double duf_du = theta * 1.0 / r * (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0)
    + x * theta * 1.0 / r * ((k2 * x * theta3 * 1.0 / r * 4.0)
        / (x2 + y2 + 1.0)+ (k3 * x * theta5 * 1.0 / r * 6.0)
        / (x2 + y2 + 1.0)+ (k4 * x * theta7 * 1.0 / r * 8.0)
        / (x2 + y2 + 1.0)+ (k1 * x * theta * 1.0 / r * 2.0)
        / (x2 + y2 + 1.0))+ ((x2)
        * (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0))
    / ((x2 + y2) * (x2 + y2 + 1.0))
    - (x2) * theta * 1.0 / pow(x2 + y2, 3.0 / 2.0)
    * (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0);

    const double duf_dv = x * theta * 1.0 / r *((k2 * y * theta3 * 1.0 / r * 4.0)
        / (x2 + y2 + 1.0)+ (k3 * y * theta5 * 1.0 / r * 6.0)
        / (x2 + y2 + 1.0)+ (k4 * y * theta7 * 1.0 / r * 8.0)
        / (x2 + y2 + 1.0)+ (k1 * y * theta * 1.0 / r * 2.0)
        / (x2 + y2 + 1.0))+ (x * y
        * (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0))
    / ((x2 + y2) * (x2 + y2 + 1.0))
    - x * y * theta * 1.0 / pow(x2 + y2, 3.0 / 2.0)
    * (k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + 1.0);

    const double dvf_du = y * theta * 1.0/ r* ((k2 * x * theta3 * 1.0/ r * 4.0)
        / (x2 + y2 + 1.0)+ (k3 * x * theta5 * 1.0/ r * 6.0)
        / (x2 + y2 + 1.0)+ (k4 * x * theta7 * 1.0/ r * 8.0)
        / (x2 + y2 + 1.0)+ (k1 * x * theta * 1.0/ r * 2.0)
        / (x2 + y2 + 1.0))+ (x * y
        * (k1 * theta2+ k2 * theta4+ k3 * theta6+ k4 * theta8 + 1.0))
    / ((x2 + y2) * (x2 + y2 + 1.0))
    - x * y * theta * 1.0/ pow(x2 + y2, 3.0 / 2.0)
    * (k1 * theta2+ k2 * theta4+ k3 * theta6+ k4 * theta8 + 1.0);

    const double dvf_dv = theta * 1.0/ r* (k1 * theta2+ k2 * theta4+ k3 * theta6+ k4 * theta8 + 1.0)
    + y * theta * 1.0/ r* ((k2 * y * theta3 * 1.0/ r * 4.0)
        / (x2 + y2 + 1.0)+ (k3 * y * theta5* 1.0 / r * 6.0)
        / (x2 + y2 + 1.0)+ (k4 * y * theta7* 1.0 / r * 8.0)
        / (x2 + y2 + 1.0)+ (k1 * y * theta * 1.0/ r * 2.0)
        / (x2 + y2 + 1.0))+ ((y2)
        * (k1 * theta2+ k2 * theta4+ k3 * theta6+ k4 * theta8 + 1.0))
    / ((x2 + y2) * (x2 + y2 + 1.0))
    - (y2) * theta * 1.0/ pow(x2 + y2, 3.0 / 2.0)
    * (k1 * theta2+ k2 * theta4+ k3 * theta6+ k4 * theta8 + 1.0);

    *out_jacobian << duf_du, duf_dv,
                     dvf_du, dvf_dv;
  }

  double scaling = (r > 1e-8) ? thetad / r : 1.0;
  x *= scaling;
  y *= scaling;
}

void EquidistantDistortion::distortParameterJacobian(
    const Eigen::VectorXd& dist_coeffs,
    const Eigen::Vector2d& point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const {
  CHECK_EQ(dist_coeffs.size(), kNumOfParams) << "dist_coeffs: invalid size!";
  CHECK_NOTNULL(out_jacobian);

  const double& x = point(0);
  const double& y = point(1);

  double r = sqrt(x*x + y*y);
  double theta = atan(r);

  const double duf_dk1 = x * pow(theta, 3.0) * 1.0 / r;
  const double duf_dk2 = x * pow(theta, 5.0) * 1.0 / r;
  const double duf_dk3 = x * pow(theta, 7.0) * 1.0 / r;
  const double duf_dk4 = x * pow(theta, 9.0) * 1.0 / r;

  const double dvf_dk1 = y * pow(theta, 3.0) * 1.0 / r;
  const double dvf_dk2 = y * pow(theta, 5.0) * 1.0 / r;
  const double dvf_dk3 = y * pow(theta, 7.0) * 1.0 / r;
  const double dvf_dk4 = y * pow(theta, 9.0) * 1.0 / r;

  out_jacobian->resize(2, kNumOfParams);
  *out_jacobian << duf_dk1, duf_dk2, duf_dk3, duf_dk4,
                   dvf_dk1, dvf_dk2, dvf_dk3, dvf_dk4;
}

void EquidistantDistortion::undistortUsingExternalCoefficients(const Eigen::VectorXd& dist_coeffs,
                                                               Eigen::Vector2d* point) const {
  CHECK_EQ(dist_coeffs.size(), kNumOfParams) << "dist_coeffs: invalid size!";
  CHECK_NOTNULL(point);

  const int n = 30;  // Max. number of iterations

  const double& k1 = dist_coeffs(0);
  const double& k2 = dist_coeffs(1);
  const double& k3 = dist_coeffs(2);
  const double& k4 = dist_coeffs(3);

  double& x = (*point)(0);
  double& y = (*point)(1);

  double theta, theta2, theta4, theta6, theta8, thetad, scaling;

  thetad = sqrt(x*x + y*y);
  theta = thetad;  // initial guess
  for (int i = n; i > 0; i--) {
    theta2 = theta * theta;
    theta4 = theta2 * theta2;
    theta6 = theta4 * theta2;
    theta8 = theta4 * theta4;
    theta = thetad / (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
  }

  scaling = tan(theta) / thetad;
  x *= scaling;
  y *= scaling;
}

bool EquidistantDistortion::distortionParametersValid(const Eigen::VectorXd& dist_coeffs) const {
  CHECK_EQ(dist_coeffs.size(), kNumOfParams) << "Invalid number of distortion coefficients (found "
        << dist_coeffs.size() << ", expected "<< kNumOfParams << ").";

  // Just check the vector size.
  return true;
}

void EquidistantDistortion::printParameters(std::ostream& out, const std::string& text) const {
  Eigen::VectorXd distortion_coefficients = getParameters();
  CHECK_EQ(distortion_coefficients.size(), kNumOfParams) << "dist_coeffs: invalid size!";

  out << text << std::endl;
  out << "Distortion: (EquidistantDistortion) " << std::endl;
  out << "  k1: " << distortion_coefficients(0) << std::endl;
  out << "  k2: " << distortion_coefficients(1) << std::endl;
  out << "  k3: " << distortion_coefficients(2) << std::endl;
  out << "  k4: " << distortion_coefficients(3) << std::endl;
}

} // namespace aslam
