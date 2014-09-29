#include <memory>
#include <utility>

#include <aslam/cameras/camera-pinhole.h>
#include <aslam/common/types.h>
#include <aslam/common/undistort-helpers.h>
#include <aslam/pipeline/undistorter-mapped.h>

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
    : Base(Eigen::Vector4d::Zero()) {
  setImageWidth(0);
  setImageHeight(0);
}

PinholeCamera::PinholeCamera(const Eigen::VectorXd& intrinsics,
                             uint32_t image_width, uint32_t image_height,
                             aslam::Distortion::UniquePtr& distortion)
  : Base(intrinsics, distortion) {
  CHECK(intrinsicsValid(intrinsics));

  setImageWidth(image_width);
  setImageHeight(image_height);
}

PinholeCamera::PinholeCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width,
                             uint32_t image_height)
    : Base(intrinsics) {
  CHECK(intrinsicsValid(intrinsics));

  setImageWidth(image_width);
  setImageHeight(image_height);
}

PinholeCamera::PinholeCamera(double focallength_cols, double focallength_rows,
                             double imagecenter_cols, double imagecenter_rows, uint32_t image_width,
                             uint32_t image_height, aslam::Distortion::UniquePtr& distortion)
    : PinholeCamera(
        Eigen::Vector4d(focallength_cols, focallength_rows, imagecenter_cols, imagecenter_rows),
        image_width, image_height, distortion) {}

PinholeCamera::PinholeCamera(double focallength_cols, double focallength_rows,
                             double imagecenter_cols, double imagecenter_rows, uint32_t image_width,
                             uint32_t image_height)
    : PinholeCamera(
        Eigen::Vector4d(focallength_cols, focallength_rows, imagecenter_cols, imagecenter_rows),
        image_width, image_height) {}

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

  return true;
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
    const Eigen::VectorXd* intrinsics_external,
    const Eigen::VectorXd* distortion_coefficients_external,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const {
  CHECK_NOTNULL(out_keypoint);

  // Determine the parameter source. (if nullptr, use internal)
  const Eigen::VectorXd* intrinsics;
  if (!intrinsics_external)
    intrinsics = &getParameters();
  else
    intrinsics = intrinsics_external;
  CHECK_EQ(intrinsics->size(), kNumOfParams) << "intrinsics: invalid size!";

  const Eigen::VectorXd* distortion_coefficients;
  if(!distortion_coefficients_external && distortion_) {
    distortion_coefficients = &getDistortion()->getParameters();
  } else {
    distortion_coefficients = distortion_coefficients_external;
  }

  const double& fu = (*intrinsics)[0];
  const double& fv = (*intrinsics)[1];
  const double& cu = (*intrinsics)[2];
  const double& cv = (*intrinsics)[3];

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
    distortion_->distortUsingExternalCoefficients(distortion_coefficients,
                                                  out_keypoint,
                                                  &J_distortion);
  } else if (distortion_) {
    // Distortion active but Jacobian NOT wanted.
    distortion_->distortUsingExternalCoefficients(distortion_coefficients,
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
    distortion_->distortParameterJacobian(distortion_coefficients_external,
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

std::unique_ptr<MappedUndistorter> PinholeCamera::createMappedUndistorter(
    float alpha, float scale, aslam::InterpolationMethod interpolation_type) const {

  CHECK_GE(alpha, 0.0); CHECK_LE(alpha, 1.0);
  CHECK_GT(scale, 0.0);

  // Only remove distortion effects.
  const bool undistort_to_pinhole = false;

  // Create a copy of the input camera (=this)
  PinholeCamera::Ptr input_camera(dynamic_cast<PinholeCamera*>(this->clone()));
  CHECK(input_camera);

  // Create the scaled output camera with removed distortion.
  Eigen::Matrix3d output_camera_matrix = common::getOptimalNewCameraMatrix(*input_camera, alpha,
                                                                           scale,
                                                                           undistort_to_pinhole);

  Eigen::Vector4d intrinsics;
  intrinsics <<  output_camera_matrix(0, 0), output_camera_matrix(1, 1),
                 output_camera_matrix(0, 2), output_camera_matrix(1, 2);

  const int output_width = static_cast<int>(scale * imageWidth());
  const int output_height = static_cast<int>(scale * imageHeight());
  PinholeCamera::Ptr output_camera = aslam::createCamera<PinholeCamera>(intrinsics, output_width,
                                                                        output_height);
  CHECK(output_camera);

  cv::Mat map_u, map_v;
  aslam::common::buildUndistortMap(*input_camera, *output_camera, undistort_to_pinhole, CV_16SC2,
                                   map_u, map_v);

  return std::unique_ptr<MappedUndistorter>(
      new MappedUndistorter(input_camera, output_camera, map_u, map_v, interpolation_type));
}

bool PinholeCamera::intrinsicsValid(const Eigen::VectorXd& intrinsics) {
  return (intrinsics.size() == parameterCount()) &&
         (intrinsics[0] > 0.0)  && //fu
         (intrinsics[1] > 0.0)  && //fv
         (intrinsics[2] > 0.0)  && //cu
         (intrinsics[3] > 0.0);    //cv
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
