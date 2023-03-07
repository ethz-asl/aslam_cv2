#include <memory>
#include <utility>

#include <aslam/cameras/camera-generic.h>

#include <aslam/cameras/camera-factory.h>
#include <aslam/common/types.h>

#include "aslam/cameras/random-camera-generator.h"

namespace aslam {
std::ostream& operator<<(std::ostream& out, const GenericCamera& camera) {
  camera.printParameters(out, std::string(""));
  return out;
}

GenericCamera::GenericCamera()
    : Base(Eigen::Matrix<double, 6, 1>::Zero(), 0, 0, Camera::Type::kGeneric) {}

GenericCamera::GenericCamera(const Eigen::VectorXd& intrinsics,
                             uint32_t image_width, uint32_t image_height,
                             aslam::Distortion::UniquePtr& distortion)
  : GenericCamera(intrinsics, image_width, image_height){
  if(distortion->getType() != Distortion::Type::kNoDistortion){
    LOG(ERROR) << "Constructed Generic Camera with Distortion, distortion set to kNoDistortion";
  }
}

GenericCamera::GenericCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width,
                             uint32_t image_height)
    : Base(intrinsics, image_width, image_height, Camera::Type::kGeneric) {
  CHECK(intrinsicsValid(intrinsics));
}

GenericCamera::GenericCamera(double calibration_min_x, double calibration_min_y,
                            double calibration_max_x, double calibration_max_y,
                            double grid_width, double grid_height, uint32_t image_width,
                             uint32_t image_height, aslam::Distortion::UniquePtr& distortion)
   : GenericCamera(
        // The use of std::vector is only a workaround to get the values into an eigen vector
        Eigen::Matrix<double, 6, 1>(std::vector<double>{calibration_min_x, calibration_min_y, calibration_max_x, 
        calibration_max_y, grid_width, grid_height}.data()),
        image_width, image_height, distortion) {} 


GenericCamera::GenericCamera(double calibration_min_x, double calibration_min_y,
                            double calibration_max_x, double calibration_max_y,
                            double grid_width, double grid_height,
                            uint32_t image_width, uint32_t image_height)
    : GenericCamera(
      // The use of std::vector is only a workaround to get the values into an eigen vector
        Eigen::Matrix<double, 6, 1>(std::vector<double>{calibration_min_x, calibration_min_y,
                calibration_max_x, calibration_max_y,
                grid_width, grid_height}.data()),
        image_width, image_height) {}

bool GenericCamera::backProject3(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                                 Eigen::Vector3d* out_point_3d) const {
  CHECK_NOTNULL(out_point_3d);
  if(!isInCalibratedArea(keypoint)){
    return false;
  }
  Eigen::Vector2d keypointGrid = transformImagePixelToGridPoint(keypoint);
  interpolateCubicBSplineSurface(keypointGrid, out_point_3d);
  *out_point_3d = (*out_point_3d).normalized();
  return true;
}


// TODO(beni) implement
const ProjectionResult GenericCamera::project3Functional(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
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
  if(!distortion_coefficients_external) {
    distortion_coefficients = &getDistortion().getParameters();
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
  if(out_jacobian_distortion) {
    // Calculate the Jacobian w.r.t to the distortion parameters,
    // if requested (and distortion set).
    distortion_->distortParameterJacobian(distortion_coefficients,
                                          *out_keypoint,
                                          out_jacobian_distortion);
    out_jacobian_distortion->row(0) *= fu;
    out_jacobian_distortion->row(1) *= fv;
  }

  if(out_jacobian_point) {
    // Distortion active and we want the Jacobian.
    distortion_->distortUsingExternalCoefficients(distortion_coefficients,
                                                  out_keypoint,
                                                  &J_distortion);
  } else {
    // Distortion active but Jacobian NOT wanted.
    distortion_->distortUsingExternalCoefficients(distortion_coefficients,
                                                  out_keypoint,
                                                  nullptr);
  }

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

  // Normalized image plane to camera plane.
  (*out_keypoint)[0] = fu * (*out_keypoint)[0] + cu;
  (*out_keypoint)[1] = fv * (*out_keypoint)[1] + cv;

  return evaluateProjectionResult(*out_keypoint, point_3d);
}

Eigen::Vector2d GenericCamera::createRandomKeypoint() const {
  Eigen::Vector2d out;
  out.setRandom();
  // Unit tests often fail when the point is near the border. Keep the point
  // away from the border.
  double border = std::min(imageWidth(), imageHeight()) * 0.1;

  out(0) = border + std::abs(out(0)) * (imageWidth() - border * 2.0);
  out(1) = border + std::abs(out(1)) * (imageHeight() - border * 2.0);

  return out;
}

Eigen::Vector3d GenericCamera::createRandomVisiblePoint(double depth) const {
  CHECK_GT(depth, 0.0) << "Depth needs to be positive!";
  Eigen::Vector3d point_3d;

  Eigen::Vector2d y = createRandomKeypoint();
  backProject3(y, &point_3d);
  point_3d /= point_3d.norm();

  // Muck with the depth. This doesn't change the pointing direction.
  return point_3d * depth;
}

bool GenericCamera::areParametersValid(const Eigen::VectorXd& parameters) {
  // TODO(beni) -> check other repo for valid constraints
  /*return (parameters.size() == parameterCount()) &&
         (parameters[0] > 0.0)  && //fu
         (parameters[1] > 0.0)  && //fv
         (parameters[2] > 0.0)  && //cu
         (parameters[3] > 0.0);    //cv*/
         return true; 
}

bool GenericCamera::intrinsicsValid(const Eigen::VectorXd& intrinsics) const {
  return areParametersValid(intrinsics);
}

// TODO(beni)
void GenericCamera::printParameters(std::ostream& out, const std::string& text) const {
  Camera::printParameters(out, text);
  /*out << "  focal length (cols,rows): "
      << fu() << ", " << fv() << std::endl;
  out << "  optical center (cols,rows): "
      << cu() << ", " << cv() << std::endl;*/

  out << "  distortion: ";
  distortion_->printParameters(out, text);
}
const double GenericCamera::kMinimumDepth = 1e-10;

bool GenericCamera::isValidImpl() const {
  return intrinsicsValid(intrinsics_);
}

// TODO(beni) add random grid
void GenericCamera::setRandomImpl() {
  GenericCamera::Ptr test_camera = GenericCamera::createTestCamera();
  CHECK(test_camera);
  line_delay_nanoseconds_ = test_camera->line_delay_nanoseconds_;
  image_width_ = test_camera->image_width_;
  image_height_ = test_camera->image_height_;
  mask_= test_camera->mask_;
  intrinsics_ = test_camera->intrinsics_;
  camera_type_ = test_camera->camera_type_;
  if (test_camera->distortion_) {
    distortion_ = std::move(test_camera->distortion_);
  }
}

// TODO(beni) check that grids are equal
bool GenericCamera::isEqualImpl(const Sensor& other, const bool verbose) const {
  const GenericCamera* other_camera =
      dynamic_cast<const GenericCamera*>(&other);
  if (other_camera == nullptr) {
    return false;
  }

  // Verify that the base members are equal.
  if (!isEqualCameraImpl(*other_camera, verbose)) {
    return false;
  }

  // Compare the distortion model (if distortion is set for both).
  if (!(*(this->distortion_) == *(other_camera->distortion_))) {
    return false;
  }

  return true;
}

// TODO(beni) add grid
GenericCamera::Ptr GenericCamera::createTestCamera() {
  GenericCamera::Ptr camera(new GenericCamera(15, 15, 736, 464, 16, 11, 640, 480));
  CameraId id;
  generateId(&id);
  camera->setId(id);
  return camera;
}

bool GenericCamera::isInCalibratedArea(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const {
  double x = keypoint.x();
  double y = keypoint.y();
  return x >= calibrationMinX() && y >= calibrationMinY() && x < calibrationMaxX() + 1 && y < calibrationMaxY() + 1;
}

Eigen::Vector2d GenericCamera::transformImagePixelToGridPoint(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const {
  return Eigen::Vector2d(
    1. + (gridWidth() - 3.) * (keypoint.x() - calibrationMinX()) / (calibrationMaxX() - calibrationMinX() + 1.),
    1. + (gridHeight() - 3.) * (keypoint.y() - calibrationMinY()) / (calibrationMaxY() - calibrationMinY() + 1.)
  );
}

Eigen::Vector2d GenericCamera::transformGridPointToImagePixel(const Eigen::Vector2d& gridpoint) const {
  return Eigen::Vector2d(
    calibrationMinX() + (gridpoint.x()-1.)*(calibrationMaxX() - calibrationMinX() + 1.)/(gridWidth() - 3.),
    calibrationMinY() + (gridpoint.y()-1.)*(calibrationMaxY() - calibrationMinY() + 1.)/(gridHeight() - 3.)
  );
}

Eigen::Vector3d GenericCamera::valueAtGridpoint(const Eigen::Vector2d gridpoint) const {
  return grid_[gridpoint.y()][gridpoint.x()];
}

void GenericCamera::interpolateCubicBSplineSurface(Eigen::Vector2d keypoint, Eigen::Vector3d* out_point_3d) const {
  CHECK_NOTNULL(out_point_3d);

  double x = keypoint.x();
  double y = keypoint.y();

  // start at bottom right corner of used 4x4 grid
  x+=2;
  y+=2;

  double floor_x = std::floor(x);
  double floor_y = std::floor(y);

  double frac_x = x - (floor_x - 3);
  double frac_y = y - (floor_y - 3);

  // i == 0
  double a_coeff = -1./6. * (frac_x - 4) * (frac_x - 4) * (frac_x - 4);

  // i == 1
  double b_coeff = 1./2. * frac_x * frac_x * frac_x - 11./2. * frac_x * frac_x + (39./2.) * frac_x - 131./6.;

  // i == 2
  double c_coeff = -1./2. * frac_x * frac_x * frac_x + 5 * frac_x * frac_x - 16 * frac_x + 50./3.;

  // i == 3
  double d_coeff = 1./6. * (frac_x - 3) * (frac_x - 3) * (frac_x - 3);


  int row0 = floor_y - 3;
  Eigen::Matrix<double, 3, 1> a = a_coeff * grid_[row0][floor_x - 3] + b_coeff * grid_[row0][floor_x - 2] 
                              + c_coeff * grid_[row0][floor_x - 1] + d_coeff * grid_[row0][floor_x];

  int row1 = floor_y - 2;
  Eigen::Matrix<double, 3, 1> b = a_coeff * grid_[row1][floor_x - 3] + b_coeff * grid_[row1][floor_x - 2] 
                              + c_coeff * grid_[row1][floor_x - 1] + d_coeff * grid_[row1][floor_x];
  int row2 = floor_y - 1;
  Eigen::Matrix<double, 3, 1> c = a_coeff * grid_[row2][floor_x - 3] + b_coeff * grid_[row2][floor_x - 2] 
                              + c_coeff * grid_[row2][floor_x - 1] + d_coeff * grid_[row2][floor_x];
  int row3 = floor_y;
  Eigen::Matrix<double, 3, 1> d = a_coeff * grid_[row3][floor_x - 3] + b_coeff * grid_[row3][floor_x - 2] 
                              + c_coeff * grid_[row3][floor_x - 1] + d_coeff * grid_[row3][floor_x];

  interpolateCubicBSpline(a, b, c, d, frac_y, out_point_3d);
}


void GenericCamera::interpolateCubicBSpline(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c, Eigen::Vector3d d, double frac_y, Eigen::Vector3d* out_point_3d) const {
  CHECK_NOTNULL(out_point_3d);

  // i == 0
  double a_coeff = -1./6. * (frac_y - 4) * (frac_y - 4) * (frac_y - 4);

  // i == 1
  double b_coeff = 1./2. * frac_y * frac_y * frac_y - 11./2. * frac_y * frac_y + (39./2.) * frac_y - 131./6.;

  // i == 2
  double c_coeff = -1./2. * frac_y * frac_y * frac_y + 5 * frac_y * frac_y - 16 * frac_y + 50./3.;

  // i == 3
  double d_coeff = 1./6. * (frac_y - 3) * (frac_y - 3) * (frac_y - 3);

  *out_point_3d = a_coeff * a + b_coeff * b + c_coeff * c + d_coeff * d;
}

bool GenericCamera::loadFromYamlNodeImpl(const YAML::Node& yaml_node) {
  if (!yaml_node.IsMap()) {
    LOG(ERROR) << "Unable to parse the camera because the node is not a map.";
    return false;
  }

  // TODO(beni) warn if distortion is supplied
  // Distortion is included into Generic Camera Model
  distortion_.reset(new aslam::NullDistortion());

  // Get the image width and height
  if (!YAML::safeGet(yaml_node, "image_width", &image_width_) ||
      !YAML::safeGet(yaml_node, "image_height", &image_height_)) {
    LOG(ERROR) << "Unable to get the image size.";
    return false;
  }

  // Get the camera type
  std::string camera_type;
  if (!YAML::safeGet(yaml_node, "type", &camera_type)) {
    LOG(ERROR) << "Unable to get camera type";
    return false;
  }
  if(camera_type != "generic"){
     LOG(ERROR) << "Camera model: \"" << camera_type << "\", but generic expected";
    return false;
  }
  camera_type_ = Type::kGeneric;

  // Get the camera intrinsics
  if (!YAML::safeGet(yaml_node, "intrinsics", &intrinsics_)) {
    LOG(ERROR) << "Unable to get intrinsics";
    return false;
  }

  if (!intrinsicsValid(intrinsics_)) {
    LOG(ERROR) << "Invalid intrinsics parameters for the " << camera_type
               << " camera model" << intrinsics_.transpose() << std::endl;
    return false;
  }

  // Get the grid for the generic model
  Eigen::Matrix<double, Eigen::Dynamic, 3> tempGrid;
  if (!YAML::safeGet(yaml_node, "grid", &tempGrid)) {
    LOG(ERROR) << "Unable to get grid";
    return false;
  }

  // Move the grid into the 2 dimensional std::vector
  grid_.resize(gridHeight(), std::vector<Eigen::Vector3d>(gridWidth()));
  for (int i = 0; i < this->gridHeight(); i++ ){
    for (int j = 0; j < this->gridWidth(); j++){
      grid_[i][j] = tempGrid.row(i*this->gridWidth() + j);
    }
  }

  // Get the optional linedelay in nanoseconds or set the default
  if (!YAML::safeGet(
          yaml_node, "line-delay-nanoseconds", &line_delay_nanoseconds_)) {
    LOG(WARNING) << "Unable to parse parameter line-delay-nanoseconds."
                 << "Setting to default value = 0.";
    line_delay_nanoseconds_ = 0;
  }

  // Get the optional compressed definition for images or set the default
  if (YAML::hasKey(yaml_node, "compressed")) {
    if (!YAML::safeGet(yaml_node, "compressed", &is_compressed_)) {
      LOG(WARNING) << "Unable to parse parameter compressed."
                   << "Setting to default value = false.";
      is_compressed_ = false;
    }
  }
  
  return true;
}

void GenericCamera::saveToYamlNodeImpl(YAML::Node* yaml_node) const { 
  CHECK_NOTNULL(yaml_node);
  YAML::Node& node = *yaml_node;

  node["compressed"] = hasCompressedImages();
  node["line-delay-nanoseconds"] = getLineDelayNanoSeconds();
  node["image_height"] = imageHeight();
  node["image_width"] = imageWidth();
  node["type"] = "generic";

  node["intrinsics"] = getParameters();
  // TODO(beni) save grid
 
}
}  // namespace aslam
