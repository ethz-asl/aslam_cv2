#include <aslam/common/eigen-helpers.h>
#include <aslam/cameras/camera-omni.h>

// TODO(slynen) Enable commented out PropertyTree support
//#include <sm/PropertyTree.hpp>

namespace aslam {
// TODO(slynen) Enable commented out PropertyTree support
//OmniCamera::OmniCamera(
//    const sm::PropertyTree & config)
//: Camera(config) {
//  _fu = config.getDouble("fu");
//  _fv = config.getDouble("fv");
//  _cu = config.getDouble("cu");
//  _cv = config.getDouble("cv");
//  imageWidth() = config.getInt("ru");
//  imageHeight() = config.getInt("rv");
//
//  //TODO(slynen): Load and instantiate correct distortion here.
//  // distortion.(config, "distortion")
//  CHECK(false) << "Loading of distortion from property tree not implemented.";
//
//}

OmniCamera::OmniCamera()
  : Camera( common::createVector5(0.0, 0.0, 0.0, 0.0, 0.0) ),
    distortion_(nullptr) {
  setImageWidth(0);
  setImageHeight(0);
}

OmniCamera::OmniCamera(const Eigen::VectorXd& intrinsics,
                       uint32_t image_width, uint32_t image_height,
                       aslam::Distortion::Ptr distortion)
: Camera( intrinsics ),
  distortion_(distortion) {
  CHECK_EQ(intrinsics.size(), kNumOfParams) << "intrinsics: invalid size!";
  setImageWidth(image_width);
  setImageHeight(image_height);
}

OmniCamera::OmniCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width,
                       uint32_t image_height)
    : OmniCamera(intrinsics, image_width, image_height, nullptr) {}

OmniCamera::OmniCamera(double xi, double focallength_cols, double focallength_rows,
                       double imagecenter_cols, double imagecenter_rows, uint32_t image_width,
                       uint32_t image_height, aslam::Distortion::Ptr distortion)
    : OmniCamera(
        common::createVector5(xi, focallength_cols, focallength_rows, imagecenter_cols,
                              imagecenter_rows), image_width, image_height, distortion) {}

OmniCamera::OmniCamera(double xi, double focallength_cols, double focallength_rows,
                       double imagecenter_cols, double imagecenter_rows, uint32_t image_width,
                       uint32_t image_height)
    : OmniCamera(xi, focallength_cols, focallength_rows, imagecenter_cols, imagecenter_rows,
                 image_width, image_height, nullptr) {}

bool OmniCamera::operator==(const Camera& other) const {
  // Check that the camera models are the same.
  const OmniCamera* rhs = dynamic_cast<const OmniCamera*>(&other);
  if (!rhs)
    return false;

  // Verify that the base members are equal.
  if (!Camera::operator==(other))
    return false;

  // Check if only one camera defines a distortion.
  if ((distortion_ && !rhs->distortion_) || (!distortion_ && rhs->distortion_))
    return false;

  // Compare the distortion model (if distortion is set for both).
  if (distortion_ && rhs->distortion_) {
    if ( !(*(this->distortion_) == *(rhs->distortion_)) )
      return false;
  }

  // Compare intrinsics parameters.
  return intrinsics_ == rhs->intrinsics_;
}


const ProjectionResult OmniCamera::project3(const Eigen::Vector3d& point_3d,
                                           Eigen::Vector2d* out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);

  // Project the point.
  const double& x = point_3d[0];
  const double& y = point_3d[1];
  const double& z = point_3d[2];

  const double d = point_3d.norm();
  const double rz = 1.0 / (z + xi() * d);

  // Check if point will lead to a valid projection
  const bool valid_proj = z > -(fov_parameter(xi()) * d);
  if(!valid_proj)
  {
    out_keypoint->setZero();
    return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
  }

  (*out_keypoint)[0] = x * rz;
  (*out_keypoint)[1] = y * rz;

  // Distort the point (if a distortion model is set)
  if (distortion_)
    distortion_->distort(out_keypoint);

  // Normalized image plane to camera plane.
  (*out_keypoint)[0] = fu() * (*out_keypoint)[0] + cu();
  (*out_keypoint)[1] = fv() * (*out_keypoint)[1] + cv();

  return evaluateProjectionResult(*out_keypoint, point_3d);
}

const ProjectionResult OmniCamera::project3(const Eigen::Vector3d& point_3d,
                                            Eigen::Vector2d* out_keypoint,
                                            Eigen::Matrix<double, 2, 3>* out_jacobian) const {
  CHECK_NOTNULL(out_keypoint);
  CHECK_NOTNULL(out_jacobian);

  const double& x = point_3d[0];
  const double& y = point_3d[1];
  const double& z = point_3d[2];

  // Normalize
  const double d = point_3d.norm();
  double rz = 1.0 / (z + xi() * d);

  // Check if point will lead to a valid projection
  const bool valid_proj = z > -(fov_parameter(xi()) * d);
  if(!valid_proj)
  {
    out_keypoint->setZero();
    return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
  }

  (*out_keypoint)[0] = x * rz;
  (*out_keypoint)[1] = y * rz;

  // Distort the point (if a distortion model is set)
  Eigen::Matrix2d J_distortion = Eigen::Matrix2d::Identity();
  if (distortion_)
    distortion_->distort(out_keypoint, &J_distortion);

  // Calculate the Jacobian w.r.t to the 3d point
  Eigen::Matrix<double, 2, 3>& J = *out_jacobian;

  rz = rz * rz / d;
  J(0, 0) = rz * (d * z + xi() * (y * y + z * z));
  J(1, 0) = -rz * xi() * x * y;
  J(0, 1) = J(1, 0);
  J(1, 1) = rz * (d * z + xi() * (x * x + z * z));
  rz = rz * (-xi() * z - d);
  J(0, 2) = x * rz;
  J(1, 2) = y * rz;
  rz = fu() * (J(0, 0) * J_distortion(0, 0) + J(1, 0) * J_distortion(0, 1));
  J(1, 0) = fv() * (J(0, 0) * J_distortion(1, 0) + J(1, 0) * J_distortion(1, 1));
  J(0, 0) = rz;
  rz = fu() * (J(0, 1) * J_distortion(0, 0) + J(1, 1) * J_distortion(0, 1));
  J(1, 1) = fv() * (J(0, 1) * J_distortion(1, 0) + J(1, 1) * J_distortion(1, 1));
  J(0, 1) = rz;
  rz = fu() * (J(0, 2) * J_distortion(0, 0) + J(1, 2) * J_distortion(0, 1));
  J(1, 2) = fv() * (J(0, 2) * J_distortion(1, 0) + J(1, 2) * J_distortion(1, 1));
  J(0, 2) = rz;

  // Normalized image plane to camera plane.
  (*out_keypoint)[0] = fu() * (*out_keypoint)[0] + cu();
  (*out_keypoint)[1] = fv() * (*out_keypoint)[1] + cv();

  return evaluateProjectionResult(*out_keypoint, point_3d);
}

bool OmniCamera::backProject3(const Eigen::Vector2d& keypoint,
                              Eigen::Vector3d* out_point_3d) const {
  CHECK_NOTNULL(out_point_3d);

  Eigen::Vector2d kp = keypoint;
  kp[0] = (kp[0] - cu()) / fu();
  kp[1] = (kp[1] - cv()) / fv();

  if(distortion_)
    distortion_->undistort(&kp);

  const double rho2_d = kp[0] * kp[0] + kp[1] * kp[1];
  const double tmpD = std::max(1 + (1 - xi()*xi()) * rho2_d, 0.0);

  (*out_point_3d)[0] = kp[0];
  (*out_point_3d)[1] = kp[1];
  (*out_point_3d)[2] = 1 - xi() * (rho2_d + 1) / (xi() + sqrt(tmpD));

  return isUndistortedKeypointValid(rho2_d, xi());
}

const ProjectionResult OmniCamera::project3Functional(
    const Eigen::Vector3d& point_3d,
    const Eigen::VectorXd& intrinsics_external,
    const Eigen::VectorXd* distortion_coefficients_external,
    Eigen::Vector2d* out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);

  // Use the external parameters.
  CHECK_EQ(intrinsics_external.size(), kNumOfParams) << "intrinsics: invalid size!";
  const double& xi = intrinsics_external[0];
  const double& fu = intrinsics_external[1];
  const double& fv = intrinsics_external[2];
  const double& cu = intrinsics_external[3];
  const double& cv = intrinsics_external[4];

  // Project the point.
  const double& x = point_3d[0];
  const double& y = point_3d[1];
  const double& z = point_3d[2];

  const double d = point_3d.norm();
  const double rz = 1.0 / (z + xi * d);

  // Check if point will lead to a valid projection
  const bool valid_proj = z > -(fov_parameter(xi) * d);
  if(!valid_proj)
  {
    out_keypoint->setZero();
    return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
  }

  (*out_keypoint)[0] = x * rz;
  (*out_keypoint)[1] = y * rz;

  // Distort the point (if a distortion model is set)
  if (distortion_)
    distortion_->distortUsingExternalCoefficients(*distortion_coefficients_external,
                                                  out_keypoint,
                                                  nullptr); // No Jacobian needed.

  // Normalized image plane to camera plane.
  (*out_keypoint)[0] = fu * (*out_keypoint)[0] + cu;
  (*out_keypoint)[1] = fv * (*out_keypoint)[1] + cv;

  return evaluateProjectionResult(*out_keypoint, point_3d);
}

const ProjectionResult OmniCamera::project3Functional(
    const Eigen::Vector3d& point_3d,
    const Eigen::VectorXd& intrinsics_external,
    const Eigen::VectorXd* distortion_coefficients_external,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const {
  CHECK_NOTNULL(out_keypoint);

  // Use the external parameters.
  CHECK_EQ(intrinsics_external.size(), kNumOfParams) << "intrinsics: invalid size!";
  const double& xi = intrinsics_external[0];
  const double& fu = intrinsics_external[1];
  const double& fv = intrinsics_external[2];
  const double& cu = intrinsics_external[3];
  const double& cv = intrinsics_external[4];

  // Project the point.
  const double& x = point_3d[0];
  const double& y = point_3d[1];
  const double& z = point_3d[2];

  const double d = point_3d.norm();
  const double rz = 1.0 / (z + xi * d);

  // Check if point will lead to a valid projection
  const bool valid_proj = z > -(fov_parameter(xi) * d);
  if(!valid_proj)
  {
    out_keypoint->setZero();
    return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
  }

  (*out_keypoint)[0] = x * rz;
  (*out_keypoint)[1] = y * rz;

  // Distort the point and get the Jacobian wrt. keypoint.
  Eigen::Matrix2d J_distortion = Eigen::Matrix2d::Identity();
  if (distortion_ && out_jacobian_point) {
    // Distortion active and we want the Jacobian.
    distortion_->distortUsingExternalCoefficients(*distortion_coefficients_external,
                                                  out_keypoint,
                                                  &J_distortion);
  } else if (distortion_) {
    // Distortion active but Jacobian NOT wanted.
    distortion_->distortUsingExternalCoefficients(*distortion_coefficients_external,
                                                  out_keypoint,
                                                  nullptr);
  }

  // Calculate the Jacobian w.r.t to the 3d point, if requested.
  if(out_jacobian_point) {
    // Jacobian including distortion
    Eigen::Matrix<double, 2, 3>& J = *out_jacobian_point;
    double rz2 = rz * rz / d;
    J(0, 0) = rz2 * (d * z + xi * (y * y + z * z));
    J(1, 0) = -rz2 * xi * x * y;
    J(0, 1) = J(1, 0);
    J(1, 1) = rz2 * (d * z + xi * (x * x + z * z));
    rz2 = rz2 * (-xi * z - d);
    J(0, 2) = x * rz2;
    J(1, 2) = y * rz2;
    rz2 = fu * (J(0, 0) * J_distortion(0, 0) + J(1, 0) * J_distortion(0, 1));
    J(1, 0) = fv * (J(0, 0) * J_distortion(1, 0) + J(1, 0) * J_distortion(1, 1));
    J(0, 0) = rz2;
    rz2 = fu * (J(0, 1) * J_distortion(0, 0) + J(1, 1) * J_distortion(0, 1));
    J(1, 1) = fv * (J(0, 1) * J_distortion(1, 0) + J(1, 1) * J_distortion(1, 1));
    J(0, 1) = rz2;
    rz2 = fu * (J(0, 2) * J_distortion(0, 0) + J(1, 2) * J_distortion(0, 1));
    J(1, 2) = fv * (J(0, 2) * J_distortion(1, 0) + J(1, 2) * J_distortion(1, 1));
    J(0, 2) = rz2;
  }

  // Calculate the Jacobian w.r.t to the intrinsic parameters, if requested.
  if(out_jacobian_intrinsics) {
    out_jacobian_intrinsics->resize(2, kNumOfParams);
    out_jacobian_intrinsics->setZero();

    Eigen::Vector2d Jxi;
    Jxi[0] = -(*out_keypoint)[0] * d * rz;
    Jxi[1] = -(*out_keypoint)[1] * d * rz;
    J_distortion.row(0) *= fu;
    J_distortion.row(1) *= fv;
    (*out_jacobian_intrinsics).col(0) = J_distortion * Jxi;

    (*out_jacobian_intrinsics)(0, 1) = (*out_keypoint)[0];
    (*out_jacobian_intrinsics)(0, 3) = 1;
    (*out_jacobian_intrinsics)(1, 2) = (*out_keypoint)[1];
    (*out_jacobian_intrinsics)(1, 4) = 1;
  }

  // Calculate the Jacobian w.r.t to the distortion parameters, if requested (and distortion set)
  if(distortion_ && out_jacobian_distortion) {
    distortion_->distortParameterJacobian(*distortion_coefficients_external,
                                          *out_keypoint,
                                          out_jacobian_distortion);
    (*out_jacobian_distortion).row(0) *= fu;
    (*out_jacobian_distortion).row(1) *= fv;
  }

  // Normalized image plane to camera plane.
  (*out_keypoint)[0] = fu * (*out_keypoint)[0] + cu;
  (*out_keypoint)[1] = fv * (*out_keypoint)[1] + cv;

  return evaluateProjectionResult(*out_keypoint, point_3d);
}

inline const ProjectionResult OmniCamera::evaluateProjectionResult(
    const Eigen::Vector2d& keypoint,
    const Eigen::Vector3d& point_3d) const {

  const bool visibility = isKeypointVisible(keypoint);

  if (visibility && (point_3d[2] > kMinimumDepth))
    return ProjectionResult(ProjectionResult::Status::KEYPOINT_VISIBLE);
  else if (!visibility && (point_3d[2] > kMinimumDepth))
    return ProjectionResult(ProjectionResult::Status::KEYPOINT_OUTSIDE_IMAGE_BOX);
  else if (point_3d[2] < 0.0)
    return ProjectionResult(ProjectionResult::Status::POINT_BEHIND_CAMERA);
  else
    return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
}

inline bool OmniCamera::isUndistortedKeypointValid(const double& rho2_d,
                                                   const double& xi) const {
  return xi <= 1.0 || rho2_d <= (1.0 / (xi * xi - 1));
}

bool OmniCamera::isLiftable(const Eigen::Vector2d& keypoint) const {
  Eigen::Vector2d y;
  y[0] = 1.0/fu() * (keypoint[0] - cu());
  y[1] = 1.0/fv() * (keypoint[1] - cv());

  if(distortion_)
    distortion_->undistort(&y);

  // Now check if it is on the sensor
  double rho2_d = y[0] * y[0] + y[1] * y[1];
  return isUndistortedKeypointValid(rho2_d, xi());
}

void OmniCamera::setParameters(const Eigen::VectorXd& params) {
  CHECK_EQ(parameterCount(), static_cast<size_t>(params.size()));
  intrinsics_ = params;
}

Eigen::Vector2d OmniCamera::createRandomKeypoint() const {
  // This is tricky...The camera model defines a circle on the normalized image
  // plane and the projection equations don't work outside of it.
  // With some manipulation, we can see that, on the normalized image plane,
  // the edge of this circle is at u^2 + v^2 = 1/(xi^2 - 1)
  // So: this function creates keypoints inside this boundary.


  // Create a point on the normalized image plane inside the boundary.
  // This is not efficient, but it should be correct.
  const double ru = imageWidth(),
               rv = imageHeight();

  Eigen::Vector2d u(ru + 1, rv + 1);
  double one_over_xixi_m_1 = 1.0 / (xi() * xi() - 1);

  while (!isLiftable(u) || !isKeypointVisible(u) ) {
    u.setRandom();
    u = u - Eigen::Vector2d(0.5, 0.5);
    u /= u.norm();
    u *= ((double) rand() / (double) RAND_MAX) * one_over_xixi_m_1;

    // Now we run the point through distortion and projection.
    // Apply distortion
    if(distortion_)
      distortion_->distort(&u);

    u[0] = fu() * u[0] + cu();
    u[1] = fv() * u[1] + cv();
  }

  return u;
}

Eigen::Vector3d OmniCamera::createRandomVisiblePoint(double depth) const {
  CHECK_GT(depth, 0.0) << "Depth needs to be positive!";


  Eigen::Vector2d y = createRandomKeypoint();

  Eigen::Vector3d point_3d;
  bool success = backProject3(y, &point_3d);
  CHECK(success) << "backprojection of createRandomVisiblePoint was unsuccessful!";
  point_3d /= point_3d.norm();

  // Muck with the depth. This doesn't change the pointing direction.
  return point_3d * depth;
}

void OmniCamera::printParameters(std::ostream& out, const std::string& text) const {
  Camera::printParameters(out, text);
  out << "  focal length (cols,rows): "
      << fu() << ", " << fv() << std::endl;
  out << "  optical center (cols,rows): "
      << cu() << ", " << cv() << std::endl;

  if(distortion_) {
    out << "  distortion: ";
    distortion_->printParameters(out, text);
  }
}
}  // namespace aslam
