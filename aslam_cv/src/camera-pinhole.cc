#include <aslam/cameras/camera-pinhole.h>
// TODO(slynen) Enable commented out PropertyTree support
//#include <sm/PropertyTree.hpp>

namespace aslam {
// TODO(slynen) Enable commented out PropertyTree support
//PinholeCamera::PinholeCamera(
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

PinholeCamera::PinholeCamera()
  : Camera( Eigen::Vector4d::Zero() ),
    distortion_(nullptr) {
  setImageWidth(0);
  setImageHeight(0);
}

PinholeCamera::PinholeCamera(const Eigen::VectorXd& intrinsics,
                             uint32_t image_width, uint32_t image_height,
                             aslam::Distortion::Ptr distortion)
  : Camera(intrinsics),
    distortion_(distortion) {
  CHECK_EQ(intrinsics.size(), kNumOfParams) << "intrinsics: invalid size!";
  setImageWidth(image_width);
  setImageHeight(image_height);
}

PinholeCamera::PinholeCamera(const Eigen::VectorXd& intrinsics,
                             uint32_t image_width, uint32_t image_height)
    : PinholeCamera(intrinsics, image_width, image_height, nullptr) {}

PinholeCamera::PinholeCamera(double focallength_cols, double focallength_rows,
                             double imagecenter_cols, double imagecenter_rows, uint32_t image_width,
                             uint32_t image_height, aslam::Distortion::Ptr distortion)
    : PinholeCamera(Eigen::Vector4d(focallength_cols, focallength_rows,
                                    imagecenter_cols, imagecenter_rows),
                     image_width, image_height, distortion) {}

PinholeCamera::PinholeCamera(double focallength_cols, double focallength_rows,
                             double imagecenter_cols, double imagecenter_rows,
                             uint32_t image_width, uint32_t image_height)
    : PinholeCamera(focallength_cols, focallength_rows, imagecenter_cols, imagecenter_rows, image_width,
                    image_height, nullptr) {}

bool PinholeCamera::operator==(const Camera& other) const {
  // Check that the camera models are the same.
  const PinholeCamera* rhs = dynamic_cast<const PinholeCamera*>(&other);
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

const ProjectionResult PinholeCamera::project3(const Eigen::Vector3d& point_3d,
                                               Eigen::Vector2d* out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);

  // Project the point.
  const double& x = point_3d[0];
  const double& y = point_3d[1];
  const double& z = point_3d[2];

  const double rz = 1.0 / z;
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

const ProjectionResult PinholeCamera::project3(const Eigen::Vector3d& point_3d,
                                               Eigen::Vector2d* out_keypoint,
                                               Eigen::Matrix<double, 2, 3>* out_jacobian) const {
  CHECK_NOTNULL(out_keypoint);
  CHECK_NOTNULL(out_jacobian);

  const double& x = point_3d[0];
  const double& y = point_3d[1];
  const double& z = point_3d[2];

  // Normalize.
  const double rz = 1.0 / z;
  (*out_keypoint)[0] = x * rz;
  (*out_keypoint)[1] = y * rz;

  // Distort the point (if a distortion model is set)
  Eigen::Matrix2d J_distortion = Eigen::Matrix2d::Identity();
  if (distortion_)
    distortion_->distort(out_keypoint, &J_distortion);

  // Calculate the Jacobian w.r.t to the 3d point
  const double rz2 = rz * rz;

  const double duf_dx =  fu() * J_distortion(0, 0) * rz;
  const double duf_dy =  fu() * J_distortion(0, 1) * rz;
  const double duf_dz = -fu() * (x * J_distortion(0, 0) + y * J_distortion(0, 1)) * rz2;
  const double dvf_dx =  fv() * J_distortion(1, 0) * rz;
  const double dvf_dy =  fv() * J_distortion(1, 1) * rz;
  const double dvf_dz = -fv() * (x * J_distortion(1, 0) + y * J_distortion(1, 1)) * rz2;

  (*out_jacobian) << duf_dx, duf_dy, duf_dz,
                     dvf_dx, dvf_dy, dvf_dz;

  // Normalized image plane to camera plane.
  (*out_keypoint)[0] = fu() * (*out_keypoint)[0] + cu();
  (*out_keypoint)[1] = fv() * (*out_keypoint)[1] + cv();

  return evaluateProjectionResult(*out_keypoint, point_3d);
}

bool PinholeCamera::backProject3(const Eigen::Vector2d& keypoint,
                                 Eigen::Vector3d* out_point_3d) const {
  CHECK_NOTNULL(out_point_3d);

  Eigen::Vector2d kp = keypoint;
  kp[0] = (kp[0] - cu()) / fu();
  kp[1] = (kp[1] - cv()) / fv();

  if(distortion_)
    distortion_->undistort(&kp);

  (*out_point_3d)[0] = kp[0];
  (*out_point_3d)[1] = kp[1];
  (*out_point_3d)[2] = 1;

  // Always valid for the pinhole model.
  return true;
}

const ProjectionResult PinholeCamera::project3Functional(
    const Eigen::Vector3d& point_3d,
    const Eigen::VectorXd& intrinsics_external,
    const Eigen::VectorXd* distortion_coefficients_external,
    Eigen::Vector2d* out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);

  // Use the external parameters.
  CHECK_EQ(intrinsics_external.size(), kNumOfParams) << "intrinsics: invalid size!";
  const double& fu = intrinsics_external[0];
  const double& fv = intrinsics_external[1];
  const double& cu = intrinsics_external[2];
  const double& cv = intrinsics_external[3];

  // Project the point.
  const double& x = point_3d[0];
  const double& y = point_3d[1];
  const double& z = point_3d[2];

  const double rz = 1.0 / z;
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

const ProjectionResult PinholeCamera::project3Functional(
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
  const double& fu = intrinsics_external[0];
  const double& fv = intrinsics_external[1];
  const double& cu = intrinsics_external[2];
  const double& cv = intrinsics_external[3];

  // Project the point.
  const double& x = point_3d[0];
  const double& y = point_3d[1];
  const double& z = point_3d[2];

  const double rz = 1.0 / z;
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
    const double rz2 = rz * rz;

    const double duf_dx =  fu * J_distortion(0, 0) * rz;
    const double duf_dy =  fu * J_distortion(0, 1) * rz;
    const double duf_dz = -fu * (x * J_distortion(0, 0) + y * J_distortion(0, 1)) * rz2;
    const double dvf_dx =  fv * J_distortion(1, 0) * rz;
    const double dvf_dy =  fv * J_distortion(1, 1) * rz;
    const double dvf_dz = -fv * (x * J_distortion(1, 0) + y * J_distortion(1, 1)) * rz2;

    (*out_jacobian_point) << duf_dx, duf_dy, duf_dz,
                             dvf_dx, dvf_dy, dvf_dz;
  }

  // Calculate the Jacobian w.r.t to the intrinsic parameters, if requested.
  if(out_jacobian_intrinsics) {
    out_jacobian_intrinsics->resize(2, kNumOfParams);
    const double duf_dfu = (*out_keypoint)[0];
    const double duf_dfv = 0.0;
    const double duf_dcu = 1.0;
    const double duf_dcv = 0.0;
    const double dvf_dfu = 0.0;
    const double dvf_dfv = (*out_keypoint)[1];
    const double dvf_dcu = 0.0;
    const double dvf_dcv = 1.0;

    (*out_jacobian_intrinsics) << duf_dfu, duf_dfv, duf_dcu, duf_dcv,
                                  dvf_dfu, dvf_dfv, dvf_dcu, dvf_dcv;
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

inline const ProjectionResult PinholeCamera::evaluateProjectionResult(
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

void PinholeCamera::setParameters(const Eigen::VectorXd& params) {
  CHECK_EQ(parameterCount(), static_cast<size_t>(params.size()));
  intrinsics_ = params;
}

Eigen::Vector2d PinholeCamera::createRandomKeypoint() const {
  Eigen::Vector2d out;
  out.setRandom();
  out(0) = std::abs(out(0)) * imageWidth();
  out(1) = std::abs(out(1)) * imageHeight();
  return out;
}

Eigen::Vector3d PinholeCamera::createRandomVisiblePoint(double depth) const {
  CHECK_GT(depth, 0.0) << "Depth needs to be positive!";
  Eigen::Vector3d point_3d;

  Eigen::Vector2d y = createRandomKeypoint();
  backProject3(y, &point_3d);
  point_3d /= point_3d.norm();

  // Muck with the depth. This doesn't change the pointing direction.
  return point_3d * depth;
}

void PinholeCamera::getBorderRays(Eigen::MatrixXd& rays) const {
  rays.resize(4, 8);
  Eigen::Vector4d ray;
  backProject4(Eigen::Vector2d(0.0, 0.0), &ray);
  rays.col(0) = ray;
  backProject4(Eigen::Vector2d(0.0, imageHeight() * 0.5), &ray);
  rays.col(1) = ray;
  backProject4(Eigen::Vector2d(0.0, imageHeight() - 1.0), &ray);
  rays.col(2) = ray;
  backProject4(Eigen::Vector2d(imageWidth() - 1.0, 0.0), &ray);
  rays.col(3) = ray;
  backProject4(Eigen::Vector2d(imageWidth() - 1.0, imageHeight() * 0.5), &ray);
  rays.col(4) = ray;
  backProject4(Eigen::Vector2d(imageWidth() - 1.0, imageHeight() - 1.0), &ray);
  rays.col(5) = ray;
  backProject4(Eigen::Vector2d(imageWidth() * 0.5, 0.0), &ray);
  rays.col(6) = ray;
  backProject4(Eigen::Vector2d(imageWidth() * 0.5, imageHeight() - 1.0), &ray);
  rays.col(7) = ray;
}

void PinholeCamera::printParameters(std::ostream& out, const std::string& text) const {
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
