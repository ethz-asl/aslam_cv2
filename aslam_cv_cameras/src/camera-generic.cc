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
    : Base(Eigen::Matrix<double, 1, 1>::Zero(), 0, 0, Camera::Type::kGeneric) {}

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
  kNumOfParams = intrinsics.size();
}

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

bool GenericCamera::backProject3WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                                 Eigen::Vector3d* out_point_3d, Eigen::Matrix<double, 3, 2>* out_jacobian_pixel) const {
  CHECK_NOTNULL(out_point_3d);
  CHECK_NOTNULL(out_jacobian_pixel);
  if(!isInCalibratedArea(keypoint)){
    return false;
  }

  Eigen::Vector2d grid_point = transformImagePixelToGridPoint(keypoint);

  double x = grid_point.x();
  double y = grid_point.y();

  // start at bottom right grid corner
  x += 2.0;
  y += 2.0;

  double floor_x = std::floor(x);
  double floor_y = std::floor(y);

  double frac_x = x - (floor_x - 3);
  double frac_y = y - (floor_y - 3);

  Eigen::Vector3d gridInterpolationWindow[4][4];
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
      gridInterpolationWindow[i][j] = gridAccess(floor_y - 3 + i, floor_x - 3 + j);
    }
  }

  CentralGenericBSpline_Unproject_ComputeResidualAndJacobian(frac_x, frac_y, gridInterpolationWindow, out_point_3d, out_jacobian_pixel);

  (*out_jacobian_pixel)(0,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(0,0));
  (*out_jacobian_pixel)(0,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(0,1));
  (*out_jacobian_pixel)(1,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(1,0));
  (*out_jacobian_pixel)(1,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(1,1));
  (*out_jacobian_pixel)(2,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(2,0));
  (*out_jacobian_pixel)(2,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(2,1));

  return true;
}

bool GenericCamera::backProject3WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint, const Eigen::Ref<const Eigen::VectorXd>& intrinsics,
                                 Eigen::Vector3d* out_point_3d, Eigen::Matrix<double, 3, 2>* out_jacobian_pixel) const {
  CHECK_NOTNULL(out_point_3d);
  CHECK_NOTNULL(out_jacobian_pixel);
  if(!isInCalibratedArea(keypoint)){
    return false;
  }

  // Eigen::Vector2d grid_point = transformImagePixelToGridPoint(keypoint); 
  Eigen::Vector2d grid_point = Eigen::Vector2d(
    1. + (intrinsics(4) - 3.) * (keypoint.x() - intrinsics(0)) / (intrinsics(2) - intrinsics(0) + 1.),
    1. + (intrinsics(5) - 3.) * (keypoint.y() - intrinsics(1)) / (intrinsics(3) - intrinsics(1) + 1.)
  );

  double x = grid_point.x();
  double y = grid_point.y();

  // start at bottom right grid corner
  x += 2.0;
  y += 2.0;

  double floor_x = std::floor(x);
  double floor_y = std::floor(y);

  double frac_x = x - (floor_x - 3);
  double frac_y = y - (floor_y - 3);

  Eigen::Vector3d gridInterpolationWindow[4][4];
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
      // gridInterpolationWindow[i][j] = gridAccess(floor_y - 3 + i, floor_x - 3 + j);
      gridInterpolationWindow[i][j] = intrinsics.segment<3>(6 + 3*((floor_y - 3 + i)*intrinsics[4] + (floor_x - 3 + j)));
    }
  }

  CentralGenericBSpline_Unproject_ComputeResidualAndJacobian(frac_x, frac_y, gridInterpolationWindow, out_point_3d, out_jacobian_pixel);

  (*out_jacobian_pixel)(0,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(0,0));
  (*out_jacobian_pixel)(0,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(0,1));
  (*out_jacobian_pixel)(1,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(1,0));
  (*out_jacobian_pixel)(1,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(1,1));
  (*out_jacobian_pixel)(2,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(2,0));
  (*out_jacobian_pixel)(2,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(2,1));

  return true;
}

const ProjectionResult GenericCamera::project3WithInitialEstimate(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    const Eigen::VectorXd* intrinsics,
    Eigen::Vector2d* out_keypoint) const {

  // optimize projected position using the Levenberg-Marquardt method
  double epsilon = 1e-12;
  int maxIterations = 100;
  Eigen::Vector3d pointDirection = point_3d.normalized();

  double lambda = -1.0;
  for(int i = 0; i < maxIterations; i++){
    Eigen::Matrix<double, 3, 2> ddxy_dxy;

    Eigen::Vector3d bearingVector;
    // unproject with jacobian
    if(!backProject3WithJacobian(*out_keypoint, *intrinsics, &bearingVector, &ddxy_dxy)){
      return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
    }

    Eigen::Vector3d diff = bearingVector - pointDirection;
    double cost = diff.squaredNorm();

    double H_0_0 = ddxy_dxy(0, 0) * ddxy_dxy(0, 0) + ddxy_dxy(1, 0) * ddxy_dxy(1, 0) + ddxy_dxy(2, 0) * ddxy_dxy(2, 0);
    double H_0_1_and_1_0 = ddxy_dxy(0, 0) * ddxy_dxy(0, 1) + ddxy_dxy(1, 0) * ddxy_dxy(1, 1) + ddxy_dxy(2, 0) * ddxy_dxy(2, 1);
    double H_1_1 = ddxy_dxy(0, 1) * ddxy_dxy(0, 1) + ddxy_dxy(1, 1) * ddxy_dxy(1, 1) + ddxy_dxy(2, 1) * ddxy_dxy(2, 1);
    double b_0 = diff.x() * ddxy_dxy(0, 0) + diff.y() * ddxy_dxy(1, 0) + diff.z() * ddxy_dxy(2, 0);
    double b_1 = diff.x() * ddxy_dxy(0, 1) + diff.y() * ddxy_dxy(1, 1) + diff.z() * ddxy_dxy(2, 1);
  
    // change to if i == 0
    if(lambda < 0){
      constexpr double initialLambdaFactor = 0.01;
      lambda = initialLambdaFactor * 0.5 * (H_0_0 + H_1_1);
    }

    bool updateAccepted = false;
    for(int lmIteration = 0; lmIteration < 10; lmIteration++){
      double H_0_0_lm = H_0_0 + lambda;
      double H_1_1_lm = H_1_1 + lambda;

      // Solve the linear system
      double x_1 = (b_1 - H_0_1_and_1_0 / H_0_0_lm * b_0) /
                    (H_1_1_lm - H_0_1_and_1_0 * H_0_1_and_1_0 / H_0_0_lm);
      double x_0 = (b_0 - H_0_1_and_1_0 * x_1) / H_0_0_lm;

      Eigen::Vector2d testResult(
        std::max(calibrationMinX(), std::min(calibrationMaxX() + 0.999, out_keypoint->x() - x_0)),
        std::max(calibrationMinY(), std::min(calibrationMaxY() + 0.999, out_keypoint->y() - x_1))
      );

      double testCost = std::numeric_limits<double>::infinity();

      Eigen::Vector3d testDirection;
      if(backProject3(testResult, &testDirection)){
        Eigen::Vector3d testDiff = testDirection - pointDirection;
        testCost = testDiff.squaredNorm();
      }

      if(testCost < cost){
        lambda *= 0.5;
        *out_keypoint = testResult;
        updateAccepted = true;
        break;
      } else {
        lambda *= 2.;
      }
    }

    
    // TODO(beni) refactor this
    if(!updateAccepted){
      if(cost < epsilon){
        return evaluateProjectionResult(*out_keypoint, point_3d);
      }
      else{
        return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
      }
    }
    
    // cost smaller that defined epsilon tolerance, bearing vector found
    if(cost < epsilon){
      return evaluateProjectionResult(*out_keypoint, point_3d);
    }
  }
  // not found in maxinterations -> bearing vector not found
  return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
}


const ProjectionResult GenericCamera::project3Functional(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    const Eigen::VectorXd* intrinsics_external,
    const Eigen::VectorXd* distortion_coefficients_external,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const {
  CHECK_NOTNULL(out_keypoint);

  if(distortion_coefficients_external){
    LOG(FATAL) << "External distortion coefficients provided for projection with generic model. Generic models have distortion " 
    << "already included in the model, abort";
  }
  if(out_jacobian_intrinsics){
    LOG(FATAL) << "Jacobian for intrinsics can't be calculated for generic models, abort";
  }
  if(out_jacobian_distortion){
    LOG(FATAL) << "Jacobian for distortion can't be calculated for generic models, distortion is included in model, abort";
  }

  const Eigen::VectorXd* intrinsics;
  if (!intrinsics_external)
    intrinsics = &getParameters();
  else
    intrinsics = intrinsics_external;

  // initialize the projected position in the center of the calibrated area
  *out_keypoint = centerOfCalibratedArea();

  ProjectionResult result = project3WithInitialEstimate(point_3d, intrinsics, out_keypoint);

  if(out_jacobian_point != nullptr && !(result == ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID))){

    const double numericalDiffDelta = 1e-4;
    for( int dimension = 0; dimension < 3; dimension++){

      Eigen::Vector3d offsetPoint = point_3d;
      offsetPoint(dimension) += numericalDiffDelta;
      Eigen::Vector2d offsetPixel = *out_keypoint;
      if(project3WithInitialEstimate(offsetPoint, intrinsics, &offsetPixel) == ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID)){
        return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
      }
      (*out_jacobian_point)(0, dimension) = (offsetPixel.x() - out_keypoint->x()) / numericalDiffDelta;
      (*out_jacobian_point)(1, dimension) = (offsetPixel.y() - out_keypoint->y()) / numericalDiffDelta;
    }
  }
  return result;
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
  return (parameters[0] >= 0.0) &&
        (parameters[1] >= 0.0) &&
        (parameters[2] >= 0.0) &&
        (parameters[3] >= 0.0) &&
        (parameters[0] < parameters[2]) &&
        (parameters[1] < parameters[3]) &&
        (parameters[4] >= 4.0) &&
        (parameters[5] >= 4.0) &&
        (parameters.size() - 6 == parameters[4]*parameters[5]*3);;
}

bool GenericCamera::intrinsicsValid(const Eigen::VectorXd& intrinsics) const {
  return areParametersValid(intrinsics);
}

// TODO(beni)
void GenericCamera::printParameters(std::ostream& out, const std::string& text) const {
  Camera::printParameters(out, text);
  out << "  minCalibration (x,y): "
      << calibrationMinX() << ", " << calibrationMinY() << std::endl;
  out << "  maxCalibration (x,y): "
      << calibrationMaxX() << ", " << calibrationMaxY() << std::endl;
  out << "  gridWidth, gridHeight (x,y): "
      << gridWidth() << ", " << gridHeight() << std::endl;
  out << "  gridvector at gridpoint (0,0), (x,y,z): "
      << firstGridValue().transpose() << std::endl;
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
  Eigen::Matrix< double, 22, 1 > intrinsics;
  for(int i = 0; i < 22; i++) intrinsics(i) = (i+1)*(i+2);
  GenericCamera::Ptr camera(new GenericCamera(intrinsics, 640, 480));
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

double GenericCamera::pixelScaleToGridScaleX(double length) const {
  return length *((gridWidth() - 3.) / (calibrationMaxX() - calibrationMinX() + 1.));
}

double GenericCamera::pixelScaleToGridScaleY(double length) const {
  return length *((gridHeight() - 3.) / (calibrationMaxY() - calibrationMinY() + 1.));
}


/*Eigen::Vector3d GenericCamera::valueAtGridpoint(const int x, const int y) const {
  //Eigen::Matrix<double, Eigen::Dynamic, 3> grid = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3>>(const_cast<double*>(intrinsics_.data() + 6), gridWidth() * gridHeight(), 3);
  //return grid.row(y * gridWidth() + x);
  return intrinsics_.segment<3>(6 + 3*(y*gridWidth() + x));
}
*/
Eigen::Vector3d GenericCamera::gridAccess(const int row, const int col) const {
  //Eigen::Matrix<double, Eigen::Dynamic, 3> grid = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3>>(const_cast<double*>(intrinsics_.data() + 6), gridWidth() * gridHeight(), 3);
  //return grid.row(y * gridWidth() + x);
  return intrinsics_.segment<3>(6 + 3*(row*gridWidth() + col));
}

Eigen::Vector3d GenericCamera::gridAccess(const Eigen::Vector2d gridpoint) const {
  //Eigen::Matrix<double, Eigen::Dynamic, 3> grid = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3>>(const_cast<double*>(intrinsics_.data() + 6), gridWidth() * gridHeight(), 3);
  //return grid.row(y * gridWidth() + x);
  return intrinsics_.segment<3>(6 + 3*(gridpoint.y()*gridWidth() + gridpoint.x()));
}


void GenericCamera::interpolateCubicBSplineSurface(Eigen::Vector2d keypoint, Eigen::Vector3d* out_point_3d) const {
  CHECK_NOTNULL(out_point_3d);

  double x = keypoint.x();
  double y = keypoint.y();

  // start at bottom right corner of used 4x4 grid
  x+=2.0;
  y+=2.0;

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
  Eigen::Matrix<double, 3, 1> a = a_coeff * gridAccess(row0, floor_x - 3) + b_coeff * gridAccess(row0, floor_x - 2) 
                              + c_coeff * gridAccess(row0, floor_x - 1) + d_coeff * gridAccess(row0, floor_x);
  int row1 = floor_y - 2;
  Eigen::Matrix<double, 3, 1> b = a_coeff * gridAccess(row1, floor_x - 3) + b_coeff * gridAccess(row1, floor_x - 2) 
                              + c_coeff * gridAccess(row1, floor_x - 1) + d_coeff * gridAccess(row1, floor_x);
  int row2 = floor_y - 1;
  Eigen::Matrix<double, 3, 1> c = a_coeff * gridAccess(row2, floor_x - 3) + b_coeff * gridAccess(row2, floor_x - 2) 
                              + c_coeff * gridAccess(row2, floor_x - 1) + d_coeff * gridAccess(row2, floor_x);
  int row3 = floor_y;
  Eigen::Matrix<double, 3, 1> d = a_coeff * gridAccess(row3, floor_x - 3) + b_coeff * gridAccess(row3, floor_x - 2) 
                              + c_coeff * gridAccess(row3, floor_x - 1) + d_coeff * gridAccess(row3, floor_x);
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

  // Distortion is directly included in the generic camera model
  const YAML::Node& distortion_config = yaml_node["distortion"];
  if(distortion_config.IsDefined() && !distortion_config.IsNull()) {
    if(!distortion_config.IsMap()) {
      LOG(ERROR) << "Unable to parse the camera because the distortion node is "
                    "not a map.";
      return false;
    }
    std::string distortion_type;
    if(YAML::safeGet(distortion_config, "type", &distortion_type)){
      if(distortion_type != "none"){
        LOG(ERROR) << "Distortion is provided for generic model, which have the distortion " 
        << "included in the model. Distortion set to NullDistortion.";
      }
    }
  }
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
  Eigen::VectorXd tempIntrinsics;
  if (!YAML::safeGet(yaml_node, "intrinsics", &tempIntrinsics)) {
    LOG(ERROR) << "Unable to get intrinsics";
    return false;
  }

  // Get the grid for the generic model
  Eigen::VectorXd tempGrid;
  if (!YAML::safeGet(yaml_node, "grid", &tempGrid)) {
    LOG(ERROR) << "Unable to get grid";
    return false;
  }
  // Concatonate tempIntrinsics and tempGrid into one intrinsics_ vector
  intrinsics_.resize(6 + tempGrid.size());
  intrinsics_ << tempIntrinsics, Eigen::Map<Eigen::VectorXd>(tempGrid.data(), tempGrid.size()); 
  kNumOfParams = intrinsics_.size();

  if (!intrinsicsValid(intrinsics_)) {
    LOG(ERROR) << "Invalid intrinsics parameters for the " << camera_type
               << " camera model" << tempIntrinsics.transpose() << std::endl;
    return false;
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
  node["intrinsics"] = getIntrinsics();
  node["grid"] = getGrid();
}

Eigen::VectorXd GenericCamera::getIntrinsics() const{
  return intrinsics_.head(6);
}

Eigen::VectorXd GenericCamera::getGrid() const {
  return intrinsics_.tail(intrinsics_.size() - 6);
}

void GenericCamera::CentralGenericBSpline_Unproject_ComputeResidualAndJacobian(double frac_x, double frac_y, Eigen::Matrix<double, 3, 1> p[4][4], Eigen::Matrix<double, 3, 1>* result, Eigen::Matrix<double, 3, 2>* dresult_dxy) const {
  const double term0 = 0.166666666666667*frac_y;
  const double term1 = -term0 + 0.666666666666667;
  const double term2 = (frac_y - 4) * (frac_y - 4);
  const double term3 = (frac_x - 4) * (frac_x - 4);
  const double term4 = 0.166666666666667*frac_x;
  const double term5 = -term4 + 0.666666666666667;
  const double term6 = p[0][0].x()*term5;
  const double term7 = (frac_x - 3) * (frac_x - 3);
  const double term8 = term4 - 0.5;
  const double term9 = p[0][3].x()*term8;
  const double term10 = frac_x * frac_x;
  const double term11 = 0.5*frac_x*term10;
  const double term12 = 19.5*frac_x - 5.5*term10 + term11 - 21.8333333333333;
  const double term13 = -16*frac_x + 5*term10 - term11 + 16.6666666666667;
  const double term14 = p[0][1].x()*term12 + p[0][2].x()*term13 + term3*term6 + term7*term9;
  const double term15 = term14*term2;
  const double term16 = term1*term15;
  const double term17 = term0 - 0.5;
  const double term18 = (frac_y - 3) * (frac_y - 3);
  const double term19 = p[3][0].x()*term5;
  const double term20 = p[3][3].x()*term8;
  const double term21 = p[3][1].x()*term12 + p[3][2].x()*term13 + term19*term3 + term20*term7;
  const double term22 = term18*term21;
  const double term23 = term17*term22;
  const double term24 = frac_y * frac_y;
  const double term25 = 0.5*frac_y*term24;
  const double term26 = -16*frac_y + 5*term24 - term25 + 16.6666666666667;
  const double term27 = p[2][0].x()*term5;
  const double term28 = p[2][3].x()*term8;
  const double term29 = p[2][1].x()*term12 + p[2][2].x()*term13 + term27*term3 + term28*term7;
  const double term30 = term26*term29;
  const double term31 = 19.5*frac_y - 5.5*term24 + term25 - 21.8333333333333;
  const double term32 = p[1][0].x()*term5;
  const double term33 = p[1][3].x()*term8;
  const double term34 = p[1][1].x()*term12 + p[1][2].x()*term13 + term3*term32 + term33*term7;
  const double term35 = term31*term34;
  const double term36 = term16 + term23 + term30 + term35;
  const double term37 = p[0][0].y()*term5;
  const double term38 = p[0][3].y()*term8;
  const double term39 = p[0][1].y()*term12 + p[0][2].y()*term13 + term3*term37 + term38*term7;
  const double term40 = term2*term39;
  const double term41 = term1*term40;
  const double term42 = p[3][0].y()*term5;
  const double term43 = p[3][3].y()*term8;
  const double term44 = p[3][1].y()*term12 + p[3][2].y()*term13 + term3*term42 + term43*term7;
  const double term45 = term18*term44;
  const double term46 = term17*term45;
  const double term47 = p[2][0].y()*term5;
  const double term48 = p[2][3].y()*term8;
  const double term49 = p[2][1].y()*term12 + p[2][2].y()*term13 + term3*term47 + term48*term7;
  const double term50 = term26*term49;
  const double term51 = p[1][0].y()*term5;
  const double term52 = p[1][3].y()*term8;
  const double term53 = p[1][1].y()*term12 + p[1][2].y()*term13 + term3*term51 + term52*term7;
  const double term54 = term31*term53;
  const double term55 = term41 + term46 + term50 + term54;
  const double term56 = p[0][0].z()*term5;
  const double term57 = p[0][3].z()*term8;
  const double term58 = p[0][1].z()*term12 + p[0][2].z()*term13 + term3*term56 + term57*term7;
  const double term59 = term2*term58;
  const double term60 = term1*term59;
  const double term61 = p[3][0].z()*term5;
  const double term62 = p[3][3].z()*term8;
  const double term63 = p[3][1].z()*term12 + p[3][2].z()*term13 + term3*term61 + term62*term7;
  const double term64 = term18*term63;
  const double term65 = term17*term64;
  const double term66 = p[2][0].z()*term5;
  const double term67 = p[2][3].z()*term8;
  const double term68 = p[2][1].z()*term12 + p[2][2].z()*term13 + term3*term66 + term67*term7;
  const double term69 = term26*term68;
  const double term70 = p[1][0].z()*term5;
  const double term71 = p[1][3].z()*term8;
  const double term72 = p[1][1].z()*term12 + p[1][2].z()*term13 + term3*term70 + term7*term71;
  const double term73 = term31*term72;
  const double term74 = term60 + term65 + term69 + term73;
  const double term75 = (term36 * term36) + (term55 * term55) + (term74 * term74);
  const double term76 = 1. / sqrt(term75);
  const double term77 = term1*term2;
  const double term78 = 0.166666666666667*term3;
  const double term79 = 0.166666666666667*term7;
  const double term80 = 1.5*term10;
  const double term81 = -11.0*frac_x + term80 + 19.5;
  const double term82 = 10*frac_x - term80 - 16;
  const double term83 = 2*frac_x;
  const double term84 = term83 - 8;
  const double term85 = term83 - 6;
  const double term86 = term17*term18;
  const double term87 = term26*(-p[2][0].x()*term78 + p[2][1].x()*term81 + p[2][2].x()*term82 + p[2][3].x()*term79 + term27*term84 + term28*term85) + term31*(-p[1][0].x()*term78 + p[1][1].x()*term81 + p[1][2].x()*term82 + p[1][3].x()*term79 + term32*term84 + term33*term85) + term77*(-p[0][0].x()*term78 + p[0][1].x()*term81 + p[0][2].x()*term82 + p[0][3].x()*term79 + term6*term84 + term85*term9) + term86*(-p[3][0].x()*term78 + p[3][1].x()*term81 + p[3][2].x()*term82 + p[3][3].x()*term79 + term19*term84 + term20*term85);
  const double term88b = 1. / sqrt(term75);
  const double term88 = term88b * term88b * term88b;
  const double term89 = (1.0L/2.0L)*term16 + (1.0L/2.0L)*term23 + (1.0L/2.0L)*term30 + (1.0L/2.0L)*term35;
  const double term90 = (1.0L/2.0L)*term41 + (1.0L/2.0L)*term46 + (1.0L/2.0L)*term50 + (1.0L/2.0L)*term54;
  const double term91 = term26*(-p[2][0].y()*term78 + p[2][1].y()*term81 + p[2][2].y()*term82 + p[2][3].y()*term79 + term47*term84 + term48*term85) + term31*(-p[1][0].y()*term78 + p[1][1].y()*term81 + p[1][2].y()*term82 + p[1][3].y()*term79 + term51*term84 + term52*term85) + term77*(-p[0][0].y()*term78 + p[0][1].y()*term81 + p[0][2].y()*term82 + p[0][3].y()*term79 + term37*term84 + term38*term85) + term86*(-p[3][0].y()*term78 + p[3][1].y()*term81 + p[3][2].y()*term82 + p[3][3].y()*term79 + term42*term84 + term43*term85);
  const double term92 = (1.0L/2.0L)*term60 + (1.0L/2.0L)*term65 + (1.0L/2.0L)*term69 + (1.0L/2.0L)*term73;
  const double term93 = term26*(-p[2][0].z()*term78 + p[2][1].z()*term81 + p[2][2].z()*term82 + p[2][3].z()*term79 + term66*term84 + term67*term85) + term31*(-p[1][0].z()*term78 + p[1][1].z()*term81 + p[1][2].z()*term82 + p[1][3].z()*term79 + term70*term84 + term71*term85) + term77*(-p[0][0].z()*term78 + p[0][1].z()*term81 + p[0][2].z()*term82 + p[0][3].z()*term79 + term56*term84 + term57*term85) + term86*(-p[3][0].z()*term78 + p[3][1].z()*term81 + p[3][2].z()*term82 + p[3][3].z()*term79 + term61*term84 + term62*term85);
  const double term94 = 2*term88*(term87*term89 + term90*term91 + term92*term93);
  const double term95 = 1.5*term24;
  const double term96 = 10*frac_y - term95 - 16;
  const double term97 = term29*term96;
  const double term98 = -11.0*frac_y + term95 + 19.5;
  const double term99 = term34*term98;
  const double term100 = 2*frac_y;
  const double term101 = term1*(term100 - 8);
  const double term102 = term101*term14;
  const double term103 = term17*(term100 - 6);
  const double term104 = term103*term21;
  const double term105 = term49*term96;
  const double term106 = term53*term98;
  const double term107 = term101*term39;
  const double term108 = term103*term44;
  const double term109 = term68*term96;
  const double term110 = term72*term98;
  const double term111 = term101*term58;
  const double term112 = term103*term63;
  const double term113 = term88*(term89*(2*term102 + 2*term104 - 0.333333333333333*term15 + 0.333333333333333*term22 + 2*term97 + 2*term99) + term90*(2*term105 + 2*term106 + 2*term107 + 2*term108 - 0.333333333333333*term40 + 0.333333333333333*term45) + term92*(2*term109 + 2*term110 + 2*term111 + 2*term112 - 0.333333333333333*term59 + 0.333333333333333*term64));
  
  (*result)[0] = term36*term76;
  (*result)[1] = term55*term76;
  (*result)[2] = term74*term76;

  (*dresult_dxy)(0, 0) = -term36*term94 + term76*term87;
  (*dresult_dxy)(0, 1) = -term113*term36 + term76*(term102 + term104 - 0.166666666666667*term15 + 0.166666666666667*term22 + term97 + term99);
  (*dresult_dxy)(1, 0) = -term55*term94 + term76*term91;
  (*dresult_dxy)(1, 1) = -term113*term55 + term76*(term105 + term106 + term107 + term108 - 0.166666666666667*term40 + 0.166666666666667*term45);
  (*dresult_dxy)(2, 0) = -term74*term94 + term76*term93;
  (*dresult_dxy)(2, 1) = -term113*term74 + term76*(term109 + term110 + term111 + term112 - 0.166666666666667*term59 + 0.166666666666667*term64);
}
}  // namespace aslam
