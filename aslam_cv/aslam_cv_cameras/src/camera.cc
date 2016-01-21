#include <memory>

#include <glog/logging.h>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion-null.h>
#include <aslam/cameras/yaml/camera-yaml-serialization.h>
#include <aslam/common/yaml-serialization.h>

// TODO(slynen) Enable commented out PropertyTree support
//#include <sm/PropertyTree.hpp>
namespace aslam {

std::ostream& operator<< (std::ostream& out, const ProjectionResult& state) {
  std::string enum_str;
  typedef ProjectionResult::Status Status;
  switch (state.status_){
    case Status::KEYPOINT_VISIBLE:            enum_str = "KEYPOINT_VISIBLE"; break;
    case Status::KEYPOINT_OUTSIDE_IMAGE_BOX:  enum_str = "KEYPOINT_OUTSIDE_IMAGE_BOX"; break;
    case Status::POINT_BEHIND_CAMERA:         enum_str = "POINT_BEHIND_CAMERA"; break;
    case Status::PROJECTION_INVALID:          enum_str = "PROJECTION_INVALID"; break;
    default:
      case Status::UNINITIALIZED:             enum_str = "UNINITIALIZED"; break;
  }
  out << "ProjectionResult: " << enum_str << std::endl;
  return out;
}

/// Camera constructor with distortion
Camera::Camera(const Eigen::VectorXd& intrinsics, aslam::Distortion::UniquePtr& distortion,
               uint32_t image_width, uint32_t image_height, Type camera_type)
    : line_delay_nanoseconds_(0),
      label_("unnamed camera"),
      image_width_(image_width),
      image_height_(image_height),
      intrinsics_(intrinsics),
      camera_type_(camera_type),
      distortion_(std::move(distortion)) {
  CHECK_NOTNULL(distortion_.get());
}

/// Camera constructor without distortion
Camera::Camera(const Eigen::VectorXd& intrinsics, uint32_t image_width, uint32_t image_height,
               Type camera_type)
    : line_delay_nanoseconds_(0),
      label_("unnamed camera"),
      image_width_(image_width),
      image_height_(image_height),
      intrinsics_(intrinsics),
      camera_type_(camera_type),
      distortion_(new NullDistortion()) {}

void Camera::printParameters(std::ostream& out, const std::string& text) const {
  if(text.size() > 0) {
    out << text << std::endl;
  }
  out << "Camera(" << this->id_ << "): " << this->label_ << std::endl;
  out << "  line delay: " << this->line_delay_nanoseconds_ << std::endl;
  out << "  image (cols,rows): " << imageWidth() << ", " << imageHeight() << std::endl;
}

bool Camera::operator==(const Camera& other) const {
  // \TODO(slynen) should we include the id and name here?
  return (this->intrinsics_ == other.intrinsics_) &&
         (this->line_delay_nanoseconds_ == other.line_delay_nanoseconds_) &&
         (this->image_width_ == other.image_width_) &&
         (this->image_height_ == other.image_height_);
}

Camera::Ptr Camera::loadFromYaml(const std::string& yaml_file) {
  try {
    YAML::Node doc = YAML::LoadFile(yaml_file.c_str());
    return doc.as<aslam::Camera::Ptr>();
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Failed to load Camera from file " << yaml_file << " with the error: \n"
               << ex.what();
  }
  // Return nullptr in the failure case.
  return Camera::Ptr();
}

bool Camera::saveToYaml(const std::string& yaml_file) const {
  try {
    YAML::Save(*this, yaml_file);
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Failed to save camera to file " << yaml_file << " with the error: \n"
               << ex.what();
    return false;
  }
  return true;
}

const ProjectionResult Camera::project3(const Eigen::Ref<const Eigen::Vector3d>& point_3d,
                                        Eigen::Vector2d* out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);
  return project3Functional(point_3d,
                            nullptr,      // Use internal intrinsic parameters.
                            nullptr,      // Use internal distortion parameters.
                            out_keypoint,
                            nullptr,      // J_point3d not needed.
                            nullptr,      // J_intrinsic not needed.
                            nullptr);     // J_distortion not needed.
}

const ProjectionResult Camera::project3(const Eigen::Ref<const Eigen::Vector3d>& point_3d,
                                        Eigen::Vector2d* out_keypoint,
                                        Eigen::Matrix<double, 2, 3>* out_jacobian) const {
  CHECK_NOTNULL(out_keypoint);
  return project3Functional(point_3d,
                            nullptr,       // Use internal intrinsic parameters.
                            nullptr,       // Use internal distortion parameters.
                            out_keypoint,
                            out_jacobian,
                            nullptr,       // J_intrinsic not needed.
                            nullptr);      // J_distortion not needed.
}

const ProjectionResult Camera::project3Functional(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    const Eigen::VectorXd* intrinsics_external,
    const Eigen::VectorXd* distortion_coefficients_external,
    Eigen::Vector2d* out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);

  return project3Functional(point_3d,
                            intrinsics_external,
                            distortion_coefficients_external,
                            out_keypoint,
                            nullptr,
                            nullptr,
                            nullptr);
}

const ProjectionResult Camera::project4(const Eigen::Ref<const Eigen::Vector4d>& point_4d,
                                        Eigen::Vector2d* out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);

  Eigen::Vector3d point_3d;
  if (point_4d[3] < 0)
    point_3d = -point_4d.head<3>();
  else
    point_3d =  point_4d.head<3>();

  return project3(point_3d, out_keypoint);
}

const ProjectionResult Camera::project4(const Eigen::Ref<const Eigen::Vector4d>& point_4d,
                                        Eigen::Vector2d* out_keypoint,
                                        Eigen::Matrix<double, 2, 4>* out_jacobian) const {
  CHECK_NOTNULL(out_keypoint);

  Eigen::Vector3d point_3d;
  if (point_4d[3] < 0)
    point_3d = -point_4d.head<3>();
  else
    point_3d =  point_4d.head<3>();

  Eigen::Matrix<double, 2, 3> Je;
  ProjectionResult ret = project3(point_3d, out_keypoint, &Je);
  out_jacobian->setZero();
  out_jacobian->topLeftCorner<2, 3>() = Je;

  return ret;
}

bool Camera::backProject4(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                          Eigen::Vector4d* out_point4d) const {
  CHECK_NOTNULL(out_point4d);

  Eigen::Vector3d point_3d;
  bool success = backProject3(keypoint, &point_3d);
  (*out_point4d) << point_3d, 0.0;

  return success;
}

bool Camera::isProjectable3(const Eigen::Ref<const Eigen::Vector3d>& p) const {
  Eigen::Vector2d k;
  const ProjectionResult& ret = project3(p, &k);
  return ret.isKeypointVisible();
}

bool Camera::isProjectable4(const Eigen::Ref<const Eigen::Vector4d>& ph) const {
  Eigen::Vector2d k;
  const ProjectionResult& ret = project4(ph, &k);
  return ret.isKeypointVisible();
}

void Camera::project3Vectorized(
    const Eigen::Ref<const Eigen::Matrix3Xd>& points_3d, Eigen::Matrix2Xd* out_keypoints,
    std::vector<ProjectionResult>* out_results) const {
  CHECK_NOTNULL(out_keypoints);
  CHECK_NOTNULL(out_results);
  out_keypoints->resize(Eigen::NoChange, points_3d.cols());
  out_results->resize(points_3d.cols(), ProjectionResult::Status::UNINITIALIZED);
  Eigen::Vector2d projection;
  for (int i = 0; i < points_3d.cols(); ++i) {
    (*out_results)[i] = project3(points_3d.col(i), &projection);
    out_keypoints->col(i) = projection;
  }
}

void Camera::backProject3Vectorized(const Eigen::Ref<const Eigen::Matrix2Xd>& keypoints,
                                    Eigen::Matrix3Xd* out_points_3d,
                                    std::vector<unsigned char>* out_success) const {
  CHECK_NOTNULL(out_points_3d);
  CHECK_NOTNULL(out_success);
  out_points_3d->resize(Eigen::NoChange, keypoints.cols());
  out_success->resize(keypoints.cols(), false);
  Eigen::Vector3d bearing;
  for(int i = 0; i < keypoints.cols(); ++i) {
    (*out_success)[i] = backProject3(keypoints.col(i), &bearing);
    out_points_3d->col(i) = bearing;
  }
}

void Camera::setMask(const cv::Mat& mask) {
  CHECK_EQ(image_height_, static_cast<size_t>(mask.rows));
  CHECK_EQ(image_width_, static_cast<size_t>(mask.cols));
  CHECK_EQ(mask.type(), CV_8UC1);
  mask_ = mask;
}

void Camera::clearMask() {
  mask_ = cv::Mat();
}

bool Camera::hasMask() const {
  return !mask_.empty();
}

const cv::Mat& Camera::getMask() const {
  return mask_;
}

ProjectionResult::Status ProjectionResult::KEYPOINT_VISIBLE =
    ProjectionResult::Status::KEYPOINT_VISIBLE;
ProjectionResult::Status ProjectionResult::KEYPOINT_OUTSIDE_IMAGE_BOX =
    ProjectionResult::Status::KEYPOINT_OUTSIDE_IMAGE_BOX;
ProjectionResult::Status ProjectionResult::POINT_BEHIND_CAMERA =
    ProjectionResult::Status::POINT_BEHIND_CAMERA;
ProjectionResult::Status ProjectionResult::PROJECTION_INVALID =
    ProjectionResult::Status::PROJECTION_INVALID;
ProjectionResult::Status ProjectionResult::UNINITIALIZED =
    ProjectionResult::Status::UNINITIALIZED;
}  // namespace aslam

