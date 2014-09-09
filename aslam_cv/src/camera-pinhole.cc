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

PinholeCamera::PinholeCamera(double focalLengthCols, double focalLengthRows,
                             double imageCenterCols, double imageCenterRows,
                             uint32_t imageWidth, uint32_t imageHeight,
                             aslam::Distortion::Ptr distortion)
: Camera( Eigen::Vector4d(focalLengthCols, focalLengthRows, imageCenterCols, imageCenterRows) ),
  distortion_(distortion) {
  setImageWidth(imageWidth);
  setImageHeight(imageHeight);
}

PinholeCamera::PinholeCamera(double focalLengthCols, double focalLengthRows,
                             double imageCenterCols, double imageCenterRows,
                             uint32_t imageWidth, uint32_t imageHeight)
: PinholeCamera(focalLengthCols, focalLengthRows,
                imageCenterCols, imageCenterRows,
                imageWidth, imageHeight,
                nullptr) { }

bool PinholeCamera::operator==(const Camera& other) const {
  // Check that the camera models are the same.
  const PinholeCamera* rhs = dynamic_cast<const PinholeCamera*>(&other);
  if (!rhs)
    return false;

  // Verify that the base members are equal.
  if (!Camera::operator==(other))
    return false;

  // Compare the distortion model (if set).
  if (distortion_ && rhs->distortion_) {
    if ( !(*(this->distortion_) == *(rhs->distortion_)) )
      return false;
  } else {
    return false;
  }

  // Compare intrinsics parameters.
  return intrinsics_ == rhs->intrinsics_;
}

//TODO(schneith): include the distortion parameters as input?
const ProjectionState PinholeCamera::project3Functional(
    const Eigen::Vector3d& point_3d,
    Eigen::Vector2d* out_keypoint,
    const Eigen::VectorXd* intrinsics_external,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics) const {
  CHECK_NOTNULL(out_keypoint);

  // Use the internal intrinsic parameters, if no external vector is provided.
  const Eigen::VectorXd* intrinsics = &intrinsics_;
  if(intrinsics_external)
    intrinsics = intrinsics_external;
  CHECK_EQ(intrinsics->size(), kNumOfParams) << "intrinsics: invalid size!";
  const double& fu = (*intrinsics)[0];
  const double& fv = (*intrinsics)[1];
  const double& cu = (*intrinsics)[2];
  const double& cv = (*intrinsics)[3];

  // Project the point.
  const double& x = point_3d[0];
  const double& y = point_3d[1];
  const double& z = point_3d[2];

  double rz = 1.0 / z;
  (*out_keypoint)[0] = x * rz;
  (*out_keypoint)[1] = y * rz;

  // Distort the point and get the Jacobian wrt. keypoint.
  Eigen::Matrix2d J_distortion = Eigen::Matrix2d::Identity();
  if (distortion_ && out_jacobian_point) {  // Distortion active and we want the Jacobian.
    distortion_->distort(out_keypoint, &J_distortion);
  } else if (distortion_) {
    distortion_->distort(out_keypoint); // Distortion active but Jacobian NOT wanted.
  }

  // Normalized image plane to camera plane.
  (*out_keypoint)[0] = fu * (*out_keypoint)[0] + cu;
  (*out_keypoint)[1] = fv * (*out_keypoint)[1] + cv;

  // Calculate the Jacobian w.r.t to the 3d point, if requested.
  if(out_jacobian_point) {
    // Jacobian including distortion
    Eigen::Matrix<double, 2, 3>& J = *out_jacobian_point;
    double rz2 = rz * rz;

    J(0, 0) =  fu * J_distortion(0, 0) * rz;
    J(0, 1) =  fu * J_distortion(0, 1) * rz;
    J(0, 2) = -fu * (x * J_distortion(0, 0) + y * J_distortion(0, 1)) * rz2;
    J(1, 0) =  fv * J_distortion(1, 0) * rz;
    J(1, 1) =  fv * J_distortion(1, 1) * rz;
    J(1, 2) = -fv * (x * J_distortion(1, 0) + y * J_distortion(1, 1)) * rz2;
  }

  // Calculate the Jacobian w.r.t to the intrinsic parameters, if requested.
  if(out_jacobian_intrinsics) {
    CHECK(false) << "TODO(schneith): fill that in";
  }

  return evaluateProjectionState(*out_keypoint, point_3d);
}

void PinholeCamera::backProject3(
    const Eigen::Vector2d& keypoint,
    Eigen::Vector3d* out_point_3d,
    const Eigen::VectorXd* intrinsics_external) const {
  CHECK_NOTNULL(out_point_3d);

  // Use the internal intrinsic parameters, if no external vector is provided.
  const Eigen::VectorXd* intrinsics = &intrinsics_;
  if(intrinsics_external)
    intrinsics = intrinsics_external;
  CHECK_EQ(intrinsics->size(), kNumOfParams) << "intrinsics: invalid size!";
  const double& fu = (*intrinsics)[0];
  const double& fv = (*intrinsics)[1];
  const double& cu = (*intrinsics)[2];
  const double& cv = (*intrinsics)[3];

  // Backproject the point.
  Eigen::Vector2d kp = keypoint;
  kp[0] = (kp[0] - cu) / fu;
  kp[1] = (kp[1] - cv) / fv;

  distortion_->undistort(&kp);

  (*out_point_3d)[0] = kp[0];
  (*out_point_3d)[1] = kp[1];
  (*out_point_3d)[2] = 1;
}

inline const ProjectionState PinholeCamera::evaluateProjectionState(
    const Eigen::Vector2d& keypoint,
    const Eigen::Vector3d& point_3d) const {

  bool visibility = isKeypointVisible(keypoint);

  if (visibility && (point_3d[2] > kMinimumDepth))
    return ProjectionState(ProjectionState::Status_t::KEYPOINT_VISIBLE);
  else if (!visibility && (point_3d[2] > kMinimumDepth))
    return ProjectionState(ProjectionState::Status_t::KEYPOINT_OUTSIDE_IMAGE_BOX);
  else if (point_3d[2] < 0.0)
    return ProjectionState(ProjectionState::Status_t::POINT_BEHIND_CAMERA);
  else
    return ProjectionState(ProjectionState::Status_t::PROJECTION_INVALID);
}

void PinholeCamera::setParameters(const Eigen::VectorXd& params) {
  CHECK_EQ(parameterCount(), static_cast<size_t>(params.size()));
  intrinsics_ = params;
}

void PinholeCamera::printParameters(std::ostream& out, const std::string& text) {
  Camera::printParameters(out,text);
  out << "  focal length (cols,rows): "
      << fu() << ", " << fv() << std::endl;
  out << "  optical center (cols,rows): "
      << cu() << ", " << cv() << std::endl;

  if(distortion_)
  {
    out << "  distortion: ";
    distortion_->printParameters(out, text);
  }
}

}  // namespace aslam
