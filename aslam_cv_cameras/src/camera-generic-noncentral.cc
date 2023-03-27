#include <memory>
#include <utility>

#include <aslam/cameras/camera-generic-noncentral.h>

#include <aslam/cameras/camera-factory.h>
#include <aslam/common/types.h>

#include "aslam/cameras/random-camera-generator.h"

namespace aslam {
std::ostream& operator<<(std::ostream& out, const GenericNoncentralCamera& camera) {
  camera.printParameters(out, std::string(""));
  return out;
}

GenericNoncentralCamera::GenericNoncentralCamera()
    : Base(Eigen::Matrix<double, 1, 1>::Zero(), 0, 0, Camera::Type::kGenericNoncentral) {}

GenericNoncentralCamera::GenericNoncentralCamera(const Eigen::VectorXd& intrinsics,
                             uint32_t image_width, uint32_t image_height,
                             aslam::Distortion::UniquePtr& distortion)
  : GenericNoncentralCamera(intrinsics, image_width, image_height){
  if(distortion->getType() != Distortion::Type::kNoDistortion){
    LOG(ERROR) << "Constructed Generic Camera with Distortion, distortion set to kNoDistortion";
  }
}

GenericNoncentralCamera::GenericNoncentralCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width,
                             uint32_t image_height)
    : Base(intrinsics, image_width, image_height, Camera::Type::kGenericNoncentral) {
  CHECK(intrinsicsValid(intrinsics));
  kNumOfParams = intrinsics.size();
}

// TODO(beni) remove or merge backProject6 using templates
bool GenericNoncentralCamera::backProject3(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                                 Eigen::Vector3d* out_line_3d) const {
    return true;
  }
bool GenericNoncentralCamera::backProject6(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                                 Eigen::ParametrizedLine<double, 3>* out_line_3d) const {
  CHECK_NOTNULL(out_line_3d);
  if(!isInCalibratedArea(keypoint)){
    return false;
  }
  Eigen::Vector2d keypointGrid = transformImagePixelToGridPoint(keypoint);

  interpolateTwoCubicBSplineSurfaces(keypointGrid, out_line_3d);
  return true;
}

bool GenericNoncentralCamera::backProject6WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint, 
                                 Eigen::ParametrizedLine<double, 3>* out_line_3d, Eigen::Matrix<double, 6, 2>* out_jacobian_pixel) const {
  CHECK_NOTNULL(out_line_3d);
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

  Eigen::Matrix<double, 6, 1> gridInterpolationWindow[4][4];
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
     gridInterpolationWindow[i][j].topRows<3>() = directionGridAccess(floor_y - 3 + i, floor_x - 3 + j);
     gridInterpolationWindow[i][j].bottomRows<3>() = pointGridAccess(floor_y - 3 + i, floor_x - 3 + j);
    }
  }

  NoncentralGenericBSpline_Unproject_ComputeResidualAndJacobian(frac_x, frac_y, gridInterpolationWindow, out_line_3d, out_jacobian_pixel);

  (*out_jacobian_pixel)(0,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(0,0));
  (*out_jacobian_pixel)(0,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(0,1));
  (*out_jacobian_pixel)(1,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(1,0));
  (*out_jacobian_pixel)(1,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(1,1));
  (*out_jacobian_pixel)(2,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(2,0));
  (*out_jacobian_pixel)(2,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(2,1));
  (*out_jacobian_pixel)(3,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(3,0));
  (*out_jacobian_pixel)(3,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(3,1));
  (*out_jacobian_pixel)(4,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(4,0));
  (*out_jacobian_pixel)(4,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(4,1));
  (*out_jacobian_pixel)(5,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(5,0));
  (*out_jacobian_pixel)(5,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(5,1));

  return true;
}

bool GenericNoncentralCamera::backProject6WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint, const Eigen::Ref<const Eigen::VectorXd>& intrinsics,
                                 Eigen::ParametrizedLine<double, 3>* out_line_3d, Eigen::Matrix<double, 6, 2>* out_jacobian_pixel) const {
  CHECK_NOTNULL(out_line_3d);
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

  Eigen::Matrix<double, 6, 1> gridInterpolationWindow[4][4];
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
     //gridInterpolationWindow[i][j].topRows<3>() = directionGridAccess(floor_y - 3 + i, floor_x - 3 + j);
     //gridInterpolationWindow[i][j].bottomRows<3>() = pointGridAccess(floor_y - 3 + i, floor_x - 3 + j);
    gridInterpolationWindow[i][j].topRows<3>() = intrinsics.segment<3>(6 + 3*intrinsics(4)*intrinsics(5) + 3*((floor_y -3 + i)*intrinsics(4) + floor_x-3+j));     
    gridInterpolationWindow[i][j].bottomRows<3>() = intrinsics.segment<3>(6 + 3*((floor_y - 3 + i)*intrinsics(4) + floor_x - 3 + j));
    }
  }

  NoncentralGenericBSpline_Unproject_ComputeResidualAndJacobian(frac_x, frac_y, gridInterpolationWindow, out_line_3d, out_jacobian_pixel);

  (*out_jacobian_pixel)(0,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(0,0));
  (*out_jacobian_pixel)(0,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(0,1));
  (*out_jacobian_pixel)(1,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(1,0));
  (*out_jacobian_pixel)(1,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(1,1));
  (*out_jacobian_pixel)(2,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(2,0));
  (*out_jacobian_pixel)(2,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(2,1));
  (*out_jacobian_pixel)(3,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(3,0));
  (*out_jacobian_pixel)(3,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(3,1));
  (*out_jacobian_pixel)(4,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(4,0));
  (*out_jacobian_pixel)(4,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(4,1));
  (*out_jacobian_pixel)(5,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(5,0));
  (*out_jacobian_pixel)(5,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(5,1));

  return true;
}

bool GenericNoncentralCamera::backProject3WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                                 Eigen::Vector3d* out_point_3d, Eigen::Matrix<double, 3, 2>* out_jacobian_pixel) const {
  return true;
}

bool GenericNoncentralCamera::backProject3WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint, const Eigen::Ref<const Eigen::VectorXd>& intrinsics,
                                 Eigen::Vector3d* out_point_3d, Eigen::Matrix<double, 3, 2>* out_jacobian_pixel) const {
  return true;
}

const ProjectionResult GenericNoncentralCamera::project3WithInitialEstimate(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    const Eigen::VectorXd* intrinsics,
    Eigen::Vector2d* out_keypoint) const {

  // optimize projected position using the Levenberg-Marquardt method
  double epsilon = 1e-12;
  int maxIterations = 100;


  double lambda = -1.0;
  for(int i = 0; i < maxIterations; i++){

    Eigen::Matrix<double, 6, 2> dline_dxy;
    Eigen::ParametrizedLine<double, 3> bearingLine;
    // unproject with jacobian
    if(!backProject6WithJacobian(*out_keypoint, *intrinsics, &bearingLine, &dline_dxy)){
      return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
    }

    Eigen::Vector3d tangent1, tangent2;
    TangentsForDirection(bearingLine.direction(), &tangent1, &tangent2);

    // Non-squared residuals
    Eigen::Vector3d diffPointToOrigin = bearingLine.origin() - point_3d;
    double d1 = tangent1.dot(diffPointToOrigin);
    double d2 = tangent2.dot(diffPointToOrigin);

    double cost = d1*d1 + d2*d2;

    // Jacobian of residuals wrt. pixel x,y [2x2]
    Eigen::Matrix<double, 6, 3> tangentsJacobianDirection;
    TangentsJacobianWrtLineDirection(bearingLine.direction(), &tangentsJacobianDirection);

    Eigen::Matrix<double, 2, 9> d_wrt_t1_t2_origin;
    d_wrt_t1_t2_origin <<
      diffPointToOrigin.x(), diffPointToOrigin.y(), diffPointToOrigin.z(), 0, 0, 0, tangent1.x(), tangent1.y(), tangent1.z(),
      0, 0, 0, diffPointToOrigin.x(), diffPointToOrigin.y(), diffPointToOrigin.z(), tangent2.x(), tangent2.y(), tangent2.z();

    Eigen::Matrix<double, 9, 2> t1_t2_origin_wrt_xy = Eigen::Matrix<double, 9, 2>::Zero();
    t1_t2_origin_wrt_xy.block<6, 2>(0, 0) = tangentsJacobianDirection * dline_dxy.block<3, 2>(0, 0);
    t1_t2_origin_wrt_xy.block<3, 2>(6, 0) = dline_dxy.template block<3, 2>(3, 0);

    Eigen::Matrix<double, 2, 2> residuals_wrt_xy = d_wrt_t1_t2_origin * t1_t2_origin_wrt_xy;

      double H_0_0 = residuals_wrt_xy(0, 0) * residuals_wrt_xy(0, 0) + residuals_wrt_xy(1, 0) * residuals_wrt_xy(1, 0);
      double H_0_1_and_1_0 = residuals_wrt_xy(0, 0) * residuals_wrt_xy(0, 1) + residuals_wrt_xy(1, 0) * residuals_wrt_xy(1, 1);
      double H_1_1 = residuals_wrt_xy(0, 1) * residuals_wrt_xy(0, 1) + residuals_wrt_xy(1, 1) * residuals_wrt_xy(1, 1);
      double b_0 = d1 * residuals_wrt_xy(0, 0) + d2 * residuals_wrt_xy(1, 0);
      double b_1 = d1 * residuals_wrt_xy(0, 1) + d2 * residuals_wrt_xy(1, 1);
  
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

      Eigen::ParametrizedLine<double, 3> testLine;
      if(backProject6(testResult, &testLine)){
        Eigen::Vector3d testTangent1, testTangent2;
        TangentsForDirection(testLine.direction(), &testTangent1, &testTangent2);

        // Non-squared residuals
        Eigen::Vector3d testDiffPointToOrigin = testLine.origin() - point_3d;
        double test_d1 = testTangent1.dot(testDiffPointToOrigin);
        double test_d2 = testTangent2.dot(testDiffPointToOrigin);
        testCost = test_d1*test_d1 + test_d2*test_d2;
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

    if(cost < epsilon){
      return evaluateProjectionResult(*out_keypoint, point_3d);
    }
    if(!updateAccepted){
      return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
    }
  }
  // not found in maxinterations -> bearing vector not found
  return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
}

/// Treating the given direction as a vector with unit length, pointing to a
/// spot on the unit sphere, determines two right-angled tangent vectors for
/// this point of the unit sphere.
void GenericNoncentralCamera::TangentsForDirection(Eigen::Vector3d direction, Eigen::Vector3d* tangent1, Eigen::Vector3d* tangent2) const {
  *tangent1 = direction.cross((fabs(direction.x()) > 0.9f) ? Eigen::Vector3d(0,1,0) : Eigen::Vector3d(1,0,0)).normalized();
  *tangent2 = direction.cross(*tangent1);
}

void GenericNoncentralCamera::TangentsJacobianWrtLineDirection(Eigen::Vector3d direction, Eigen::Matrix<double, 6, 3>* jacobian) const {
  if (fabs(direction.x()) > 0.9f) {
    const double term0 = direction.x() * direction.x();
    const double term1 = direction.z() * direction.z();
    const double term2 = term0 + term1;
    const double term7 = 1. / sqrt(term2);
    const double term3 = term7 * term7 * term7;
    const double term4 = direction.x()*direction.z()*term3;
    const double term5 = term0*term3;
    const double term6 = term1*term3;
    const double term8 = direction.x()*term7;
    const double term9 = -direction.y()*term4;
    const double term10 = direction.z()*term7;
    
    *jacobian << term4, 0, -term5,
                 0, 0, 0,
                 term6, 0, -term4,
                 direction.y()*term6, term8, term9,
                 -term8, 0, -term10,
                 term9, term10, direction.y()*term5;
  } else {
    const double term0 = direction.y() * direction.y();
    const double term1 = direction.z() * direction.z();
    const double term2 = term0 + term1;
    const double term7 = 1. / sqrt(term2);
    const double term3 = term7 * term7 * term7;
    const double term4 = direction.y()*direction.z()*term3;
    const double term5 = term0*term3;
    const double term6 = term1*term3;
    const double term8 = direction.y()*term7;
    const double term9 = direction.z()*term7;
    const double term10 = -direction.x()*term4;
    
    *jacobian << 0, 0, 0,
                 0, -term4, term5,
                 0, -term6, term4,
                 0, -term8, -term9,
                 term8, direction.x()*term6, term10,
                 term9, term10, direction.x()*term5;
  }
}

const ProjectionResult GenericNoncentralCamera::project3Functional(
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

Eigen::Vector2d GenericNoncentralCamera::createRandomKeypoint() const {
  Eigen::Vector2d out;
  out.setRandom();
  // Unit tests often fail when the point is near the border. Keep the point
  // away from the border.
  double border = std::min(imageWidth(), imageHeight()) * 0.1;

  out(0) = border + std::abs(out(0)) * (imageWidth() - border * 2.0);
  out(1) = border + std::abs(out(1)) * (imageHeight() - border * 2.0);

  return out;
}

Eigen::Vector3d GenericNoncentralCamera::createRandomVisiblePoint(double depth) const {
  CHECK_GT(depth, 0.0) << "Depth needs to be positive!";
  Eigen::Vector3d point_3d;

  Eigen::Vector2d y = createRandomKeypoint();
  backProject3(y, &point_3d);
  point_3d /= point_3d.norm();

  // Muck with the depth. This doesn't change the pointing direction.
  return point_3d * depth;
}

bool GenericNoncentralCamera::areParametersValid(const Eigen::VectorXd& parameters) {
  return(parameters[0] >= 0.0) &&
        (parameters[1] >= 0.0) &&
        (parameters[2] >= 0.0) &&
        (parameters[3] >= 0.0) &&
        (parameters[0] < parameters[2]) &&
        (parameters[1] < parameters[3]) &&
        (parameters[4] >= 4.0) &&
        (parameters[5] >= 4.0) &&
        (parameters.size() - 6 == parameters[4]*parameters[5]*3*2);;
}

bool GenericNoncentralCamera::intrinsicsValid(const Eigen::VectorXd& intrinsics) const {
  return areParametersValid(intrinsics);
}

void GenericNoncentralCamera::printParameters(std::ostream& out, const std::string& text) const {
  Camera::printParameters(out, text);
  out << "  minCalibration (x,y): "
      << calibrationMinX() << ", " << calibrationMinY() << std::endl;
  out << "  maxCalibration (x,y): "
      << calibrationMaxX() << ", " << calibrationMaxY() << std::endl;
  out << "  gridWidth, gridHeight (x,y): "
      << gridWidth() << ", " << gridHeight() << std::endl;
  out << "  point vector at gridpoint (0,0), (x,y,z): "
      << firstPointGridValue().transpose() << std::endl;
  out << "  direction vector at gridpoint (0,0), (x,y,z): "
      << firstDirectionGridValue().transpose() << std::endl;
}
const double GenericNoncentralCamera::kMinimumDepth = 1e-10;

bool GenericNoncentralCamera::isValidImpl() const {
  return intrinsicsValid(intrinsics_);
}

void GenericNoncentralCamera::setRandomImpl() {
  GenericNoncentralCamera::Ptr test_camera = GenericNoncentralCamera::createTestCamera();
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

bool GenericNoncentralCamera::isEqualImpl(const Sensor& other, const bool verbose) const {
  const GenericNoncentralCamera* other_camera =
      dynamic_cast<const GenericNoncentralCamera*>(&other);
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

GenericNoncentralCamera::Ptr GenericNoncentralCamera::createTestCamera() {
  Eigen::Matrix< double, 6+4*4*3, 1 > intrinsics;
  intrinsics << 1, 1, 299, 199, 4, 4;
  for(int i = 0; i < 4*4; i++){
    Eigen::Vector3d tempVec = Eigen::Vector3d(i+1, i+2, i+3).normalized();
    for(int dim = 0; dim < 3; dim++){
      intrinsics(6 + 3*i + dim) = tempVec(dim);
    }
  } 
  GenericNoncentralCamera::Ptr camera(new GenericNoncentralCamera(intrinsics, 640, 480));
  CameraId id;
  generateId(&id);
  camera->setId(id);
  return camera;
}

bool GenericNoncentralCamera::isInCalibratedArea(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const {
  double x = keypoint.x();
  double y = keypoint.y();
  return x >= calibrationMinX() && y >= calibrationMinY() && x < calibrationMaxX() + 1 && y < calibrationMaxY() + 1;
}

Eigen::Vector2d GenericNoncentralCamera::transformImagePixelToGridPoint(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const {
  return Eigen::Vector2d(
    1. + (gridWidth() - 3.) * (keypoint.x() - calibrationMinX()) / (calibrationMaxX() - calibrationMinX() + 1.),
    1. + (gridHeight() - 3.) * (keypoint.y() - calibrationMinY()) / (calibrationMaxY() - calibrationMinY() + 1.)
  );
}

Eigen::Vector2d GenericNoncentralCamera::transformGridPointToImagePixel(const Eigen::Vector2d& gridpoint) const {
  return Eigen::Vector2d(
    calibrationMinX() + (gridpoint.x()-1.)*(calibrationMaxX() - calibrationMinX() + 1.)/(gridWidth() - 3.),
    calibrationMinY() + (gridpoint.y()-1.)*(calibrationMaxY() - calibrationMinY() + 1.)/(gridHeight() - 3.)
  );
}

double GenericNoncentralCamera::pixelScaleToGridScaleX(double length) const {
  return length *((gridWidth() - 3.) / (calibrationMaxX() - calibrationMinX() + 1.));
}

double GenericNoncentralCamera::pixelScaleToGridScaleY(double length) const {
  return length *((gridHeight() - 3.) / (calibrationMaxY() - calibrationMinY() + 1.));
}



Eigen::Vector3d GenericNoncentralCamera::pointGridAccess(const int row, const int col) const {
  return intrinsics_.segment<3>(6 + 3*(row*gridWidth() + col));
}

Eigen::Vector3d GenericNoncentralCamera::pointGridAccess(const Eigen::Vector2d gridpoint) const {
  return intrinsics_.segment<3>(6 + 3*(gridpoint.y()*gridWidth() + gridpoint.x()));
}

Eigen::Vector3d GenericNoncentralCamera::directionGridAccess(const int row, const int col) const {
  return intrinsics_.segment<3>(6 + 3*gridWidth()*gridHeight() + 3*(row*gridWidth() + col));
}

Eigen::Vector3d GenericNoncentralCamera::directionGridAccess(const Eigen::Vector2d gridpoint) const {
  return intrinsics_.segment<3>(6 + 3*gridWidth()*gridHeight() + 3*(gridpoint.y()*gridWidth() + gridpoint.x()));
}

void GenericNoncentralCamera::interpolateTwoCubicBSplineSurfaces(Eigen::Vector2d keypoint, Eigen::ParametrizedLine<double, 3>* out_line_3d) const {
  CHECK_NOTNULL(out_line_3d);

  double x = keypoint.x();
  double y = keypoint.y();

  // start at bottom right corner of used 4x4 grid
  x += 2.0;
  y += 2.0;

  double floor_x = std::floor(x);
  double floor_y = std::floor(y);

  double frac_x = x - (floor_x - 3);
  double frac_y = y - (floor_y - 3);

  // i == 0
  double a_coeff_x = -1./6. * (frac_x - 4) * (frac_x - 4) * (frac_x - 4);

  // i == 1
  double b_coeff_x = 1./2. * frac_x * frac_x * frac_x - 11./2. * frac_x * frac_x + (39./2.) * frac_x - 131./6.;

  // i == 2
  double c_coeff_x = -1./2. * frac_x * frac_x * frac_x + 5 * frac_x * frac_x - 16 * frac_x + 50./3.;

  // i == 3
  double d_coeff_x = 1./6. * (frac_x - 3) * (frac_x - 3) * (frac_x - 3);

  
  int row0 = floor_y - 3;
  Eigen::Matrix<double, 3, 1> a_point = a_coeff_x * pointGridAccess(row0, floor_x - 3) + b_coeff_x * pointGridAccess(row0, floor_x - 2) 
                              + c_coeff_x * pointGridAccess(row0, floor_x - 1) + d_coeff_x * pointGridAccess(row0, floor_x);
  Eigen::Matrix<double, 3, 1> a_direction = a_coeff_x * directionGridAccess(row0, floor_x - 3) + b_coeff_x * directionGridAccess(row0, floor_x - 2) 
                              + c_coeff_x * directionGridAccess(row0, floor_x - 1) + d_coeff_x * directionGridAccess(row0, floor_x);

  int row1 = floor_y - 2;
  Eigen::Matrix<double, 3, 1> b_point = a_coeff_x * pointGridAccess(row1, floor_x - 3) + b_coeff_x * pointGridAccess(row1, floor_x - 2) 
                              + c_coeff_x * pointGridAccess(row1, floor_x - 1) + d_coeff_x * pointGridAccess(row1, floor_x);
  Eigen::Matrix<double, 3, 1> b_direction= a_coeff_x * directionGridAccess(row1, floor_x - 3) + b_coeff_x * directionGridAccess(row1, floor_x - 2) 
                              + c_coeff_x * directionGridAccess(row1, floor_x - 1) + d_coeff_x * directionGridAccess(row1, floor_x);

  int row2 = floor_y - 1;
  Eigen::Matrix<double, 3, 1> c_point = a_coeff_x * pointGridAccess(row2, floor_x - 3) + b_coeff_x * pointGridAccess(row2, floor_x - 2) 
                              + c_coeff_x * pointGridAccess(row2, floor_x - 1) + d_coeff_x * pointGridAccess(row2, floor_x);
  Eigen::Matrix<double, 3, 1> c_direction = a_coeff_x * directionGridAccess(row2, floor_x - 3) + b_coeff_x * directionGridAccess(row2, floor_x - 2) 
                              + c_coeff_x * directionGridAccess(row2, floor_x - 1) + d_coeff_x * directionGridAccess(row2, floor_x);

  int row3 = floor_y;
  Eigen::Matrix<double, 3, 1> d_point = a_coeff_x * pointGridAccess(row3, floor_x - 3) + b_coeff_x * pointGridAccess(row3, floor_x - 2) 
                              + c_coeff_x * pointGridAccess(row3, floor_x - 1) + d_coeff_x * pointGridAccess(row3, floor_x);
  Eigen::Matrix<double, 3, 1> d_direction = a_coeff_x * directionGridAccess(row3, floor_x - 3) + b_coeff_x * directionGridAccess(row3, floor_x - 2) 
                              + c_coeff_x * directionGridAccess(row3, floor_x - 1) + d_coeff_x * directionGridAccess(row3, floor_x);

  // i == 0
  double a_coeff_y = -1./6. * (frac_y - 4) * (frac_y - 4) * (frac_y - 4);

  // i == 1
  double b_coeff_y = 1./2. * frac_y * frac_y * frac_y - 11./2. * frac_y * frac_y + (39./2.) * frac_y - 131./6.;

  // i == 2
  double c_coeff_y = -1./2. * frac_y * frac_y * frac_y + 5 * frac_y * frac_y - 16 * frac_y + 50./3.;

  // i == 3
  double d_coeff_y = 1./6. * (frac_y - 3) * (frac_y - 3) * (frac_y - 3);

  Eigen::Vector3d point = a_coeff_y * a_point + b_coeff_y * b_point + c_coeff_y * c_point + d_coeff_y * d_point;
  Eigen::Vector3d direction = a_coeff_y * a_direction + b_coeff_y * b_direction + c_coeff_y * c_direction + d_coeff_y * d_direction;
  *out_line_3d = Eigen::ParametrizedLine<double, 3>(point, direction.normalized());
}


void GenericNoncentralCamera::interpolateCubicBSpline(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c, Eigen::Vector3d d, double frac_y, Eigen::Vector3d* out_point_3d) const {
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

bool GenericNoncentralCamera::loadFromYamlNodeImpl(const YAML::Node& yaml_node) {
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
  if(camera_type != "generic_noncentral"){
     LOG(ERROR) << "Camera model: \"" << camera_type << "\", but generic expected";
    return false;
  }
  camera_type_ = Type::kGenericNoncentral;

  // Get the camera intrinsics
  Eigen::VectorXd tempIntrinsics;
  if (!YAML::safeGet(yaml_node, "intrinsics", &tempIntrinsics)) {
    LOG(ERROR) << "Unable to get intrinsics";
    return false;
  }

  // Get the point grid for the generic model
  Eigen::Matrix<double, Eigen::Dynamic, 3> tempPointGrid;
  if (!YAML::safeGet(yaml_node, "point_grid", &tempPointGrid)) {
    LOG(ERROR) << "Unable to get point_grid";
    return false;
  }
  // Get the directions grid for the generic model
  Eigen::Matrix<double, Eigen::Dynamic, 3> tempDirectionGrid;
  if (!YAML::safeGet(yaml_node, "direction_grid", &tempDirectionGrid)) {
    LOG(ERROR) << "Unable to get direction_grid";
    return false;
  }
  // Concatonate tempIntrinsics and tempGrid into one intrinsics_ vector
  intrinsics_.resize(6 + tempPointGrid.size() + tempDirectionGrid.size());
  intrinsics_ << tempIntrinsics; 
  for (int i = 0; i < tempPointGrid.rows(); i++){
    Eigen::Vector3d tempVec = tempPointGrid.row(i);
    for(int dim = 0; dim < 3; dim++){
      intrinsics_(6 + 3*i + dim) = tempVec(dim);
    }
  }
  for (int i = 0; i < tempDirectionGrid.rows(); i++){
    // only renormalize directions, but not the points!
    Eigen::Vector3d tempVec = tempDirectionGrid.row(i).normalized();
    for(int dim = 0; dim < 3; dim++){
      intrinsics_(6 + 3*tempIntrinsics[4]*tempIntrinsics[5] + 3*i + dim) = tempVec(dim);
    }
  }

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

void GenericNoncentralCamera::saveToYamlNodeImpl(YAML::Node* yaml_node) const { 
  CHECK_NOTNULL(yaml_node);
  YAML::Node& node = *yaml_node;

  node["compressed"] = hasCompressedImages();
  node["line-delay-nanoseconds"] = getLineDelayNanoSeconds();
  node["image_height"] = imageHeight();
  node["image_width"] = imageWidth();
  node["type"] = "generic_noncentral";
  node["intrinsics"] = getIntrinsics();
  node["grid"] = getGrid();
}

Eigen::VectorXd GenericNoncentralCamera::getIntrinsics() const{
  return intrinsics_.head(6);
}

Eigen::VectorXd GenericNoncentralCamera::getGrid() const {
  return intrinsics_.tail(intrinsics_.size() - 6);
}

void GenericNoncentralCamera::NoncentralGenericBSpline_Unproject_ComputeResidualAndJacobian(double frac_x, double frac_y, Eigen::Matrix<double, 6, 1> l[4][4], Eigen::ParametrizedLine<double, 3>* result, Eigen::Matrix<double, 6, 2>* dresult_dxy) const {
  const double term0 = 0.166666666666667*frac_y;
  const double term1 = -term0 + 0.666666666666667;
  const double term2 = (frac_y - 4) * (frac_y - 4);
  const double term3 = (frac_x - 4) * (frac_x - 4);
  const double term4 = 0.166666666666667*frac_x;
  const double term5 = -term4 + 0.666666666666667;
  const double term6 = l[0][0].x()*term5;
  const double term7 = (frac_x - 3) * (frac_x - 3);
  const double term8 = term4 - 0.5;
  const double term9 = l[0][3].x()*term8;
  const double term10 = frac_x * frac_x;
  const double term11 = 0.5*term10*frac_x;
  const double term12 = 19.5*frac_x - 5.5*term10 + term11 - 21.8333333333333;
  const double term13 = -16*frac_x + 5*term10 - term11 + 16.6666666666667;
  const double term14 = l[0][1].x()*term12 + l[0][2].x()*term13 + term3*term6 + term7*term9;
  const double term15 = term14*term2;
  const double term16 = term1*term15;
  const double term17 = term0 - 0.5;
  const double term18 = (frac_y - 3) * (frac_y - 3);
  const double term19 = l[3][0].x()*term5;
  const double term20 = l[3][3].x()*term8;
  const double term21 = l[3][1].x()*term12 + l[3][2].x()*term13 + term19*term3 + term20*term7;
  const double term22 = term18*term21;
  const double term23 = term17*term22;
  const double term24 = frac_y * frac_y;
  const double term25 = 0.5*term24*frac_y;
  const double term26 = -16*frac_y + 5*term24 - term25 + 16.6666666666667;
  const double term27 = l[2][0].x()*term5;
  const double term28 = l[2][3].x()*term8;
  const double term29 = l[2][1].x()*term12 + l[2][2].x()*term13 + term27*term3 + term28*term7;
  const double term30 = term26*term29;
  const double term31 = 19.5*frac_y - 5.5*term24 + term25 - 21.8333333333333;
  const double term32 = l[1][0].x()*term5;
  const double term33 = l[1][3].x()*term8;
  const double term34 = l[1][1].x()*term12 + l[1][2].x()*term13 + term3*term32 + term33*term7;
  const double term35 = term31*term34;
  const double term36 = term16 + term23 + term30 + term35;
  const double term37 = l[0][0].y()*term5;
  const double term38 = l[0][3].y()*term8;
  const double term39 = l[0][1].y()*term12 + l[0][2].y()*term13 + term3*term37 + term38*term7;
  const double term40 = term2*term39;
  const double term41 = term1*term40;
  const double term42 = l[3][0].y()*term5;
  const double term43 = l[3][3].y()*term8;
  const double term44 = l[3][1].y()*term12 + l[3][2].y()*term13 + term3*term42 + term43*term7;
  const double term45 = term18*term44;
  const double term46 = term17*term45;
  const double term47 = l[2][0].y()*term5;
  const double term48 = l[2][3].y()*term8;
  const double term49 = l[2][1].y()*term12 + l[2][2].y()*term13 + term3*term47 + term48*term7;
  const double term50 = term26*term49;
  const double term51 = l[1][0].y()*term5;
  const double term52 = l[1][3].y()*term8;
  const double term53 = l[1][1].y()*term12 + l[1][2].y()*term13 + term3*term51 + term52*term7;
  const double term54 = term31*term53;
  const double term55 = term41 + term46 + term50 + term54;
  const double term56 = l[0][0].z()*term5;
  const double term57 = l[0][3].z()*term8;
  const double term58 = l[0][1].z()*term12 + l[0][2].z()*term13 + term3*term56 + term57*term7;
  const double term59 = term2*term58;
  const double term60 = term1*term59;
  const double term61 = l[3][0].z()*term5;
  const double term62 = l[3][3].z()*term8;
  const double term63 = l[3][1].z()*term12 + l[3][2].z()*term13 + term3*term61 + term62*term7;
  const double term64 = term18*term63;
  const double term65 = term17*term64;
  const double term66 = l[2][0].z()*term5;
  const double term67 = l[2][3].z()*term8;
  const double term68 = l[2][1].z()*term12 + l[2][2].z()*term13 + term3*term66 + term67*term7;
  const double term69 = term26*term68;
  const double term70 = l[1][0].z()*term5;
  const double term71 = l[1][3].z()*term8;
  const double term72 = l[1][1].z()*term12 + l[1][2].z()*term13 + term3*term70 + term7*term71;
  const double term73 = term31*term72;
  const double term74 = term60 + term65 + term69 + term73;
  const double term75 = term36 * term36 + term55 * term55 + term74 * term74;
  const double term76 = 1 / sqrtf(term75);
  const double term77 = term1*term2;
  const double term78 = l[0][0](3)*term5;
  const double term79 = l[0][3](3)*term8;
  const double term80 = l[0][1](3)*term12 + l[0][2](3)*term13 + term3*term78 + term7*term79;
  const double term81 = term17*term18;
  const double term82 = l[3][0](3)*term5;
  const double term83 = l[3][3](3)*term8;
  const double term84 = l[3][1](3)*term12 + l[3][2](3)*term13 + term3*term82 + term7*term83;
  const double term85 = l[2][0](3)*term5;
  const double term86 = l[2][3](3)*term8;
  const double term87 = l[2][1](3)*term12 + l[2][2](3)*term13 + term3*term85 + term7*term86;
  const double term88 = l[1][0](3)*term5;
  const double term89 = l[1][3](3)*term8;
  const double term90 = l[1][1](3)*term12 + l[1][2](3)*term13 + term3*term88 + term7*term89;
  const double term91 = l[0][0](4)*term5;
  const double term92 = l[0][3](4)*term8;
  const double term93 = l[0][1](4)*term12 + l[0][2](4)*term13 + term3*term91 + term7*term92;
  const double term94 = l[3][0](4)*term5;
  const double term95 = l[3][3](4)*term8;
  const double term96 = l[3][1](4)*term12 + l[3][2](4)*term13 + term3*term94 + term7*term95;
  const double term97 = l[2][0](4)*term5;
  const double term98 = l[2][3](4)*term8;
  const double term99 = l[2][1](4)*term12 + l[2][2](4)*term13 + term3*term97 + term7*term98;
  const double term100 = l[1][0](4)*term5;
  const double term101 = l[1][3](4)*term8;
  const double term102 = l[1][1](4)*term12 + l[1][2](4)*term13 + term100*term3 + term101*term7;
  const double term103 = l[0][0](5)*term5;
  const double term104 = l[0][3](5)*term8;
  const double term105 = l[0][1](5)*term12 + l[0][2](5)*term13 + term103*term3 + term104*term7;
  const double term106 = l[3][0](5)*term5;
  const double term107 = l[3][3](5)*term8;
  const double term108 = l[3][1](5)*term12 + l[3][2](5)*term13 + term106*term3 + term107*term7;
  const double term109 = l[2][0](5)*term5;
  const double term110 = l[2][3](5)*term8;
  const double term111 = l[2][1](5)*term12 + l[2][2](5)*term13 + term109*term3 + term110*term7;
  const double term112 = l[1][0](5)*term5;
  const double term113 = l[1][3](5)*term8;
  const double term114 = l[1][1](5)*term12 + l[1][2](5)*term13 + term112*term3 + term113*term7;
  const double term115 = 0.166666666666667*term3;
  const double term116 = 0.166666666666667*term7;
  const double term117 = 1.5*term10;
  const double term118 = -11.0*frac_x + term117 + 19.5;
  const double term119 = 10*frac_x - term117 - 16;
  const double term120 = 2*frac_x;
  const double term121 = term120 - 8;
  const double term122 = term120 - 6;
  const double term123 = term26*(-l[2][0].x()*term115 + l[2][1].x()*term118 + l[2][2].x()*term119 + l[2][3].x()*term116 + term121*term27 + term122*term28) + term31*(-l[1][0].x()*term115 + l[1][1].x()*term118 + l[1][2].x()*term119 + l[1][3].x()*term116 + term121*term32 + term122*term33) + term77*(-l[0][0].x()*term115 + l[0][1].x()*term118 + l[0][2].x()*term119 + l[0][3].x()*term116 + term121*term6 + term122*term9) + term81*(-l[3][0].x()*term115 + l[3][1].x()*term118 + l[3][2].x()*term119 + l[3][3].x()*term116 + term121*term19 + term122*term20);
  const double term124_temp = sqrtf(term75);
  const double term124 = 1 / (term124_temp * term124_temp * term124_temp);
  const double term125 = (1.0L/2.0L)*term16 + (1.0L/2.0L)*term23 + (1.0L/2.0L)*term30 + (1.0L/2.0L)*term35;
  const double term126 = (1.0L/2.0L)*term41 + (1.0L/2.0L)*term46 + (1.0L/2.0L)*term50 + (1.0L/2.0L)*term54;
  const double term127 = term26*(-l[2][0].y()*term115 + l[2][1].y()*term118 + l[2][2].y()*term119 + l[2][3].y()*term116 + term121*term47 + term122*term48) + term31*(-l[1][0].y()*term115 + l[1][1].y()*term118 + l[1][2].y()*term119 + l[1][3].y()*term116 + term121*term51 + term122*term52) + term77*(-l[0][0].y()*term115 + l[0][1].y()*term118 + l[0][2].y()*term119 + l[0][3].y()*term116 + term121*term37 + term122*term38) + term81*(-l[3][0].y()*term115 + l[3][1].y()*term118 + l[3][2].y()*term119 + l[3][3].y()*term116 + term121*term42 + term122*term43);
  const double term128 = (1.0L/2.0L)*term60 + (1.0L/2.0L)*term65 + (1.0L/2.0L)*term69 + (1.0L/2.0L)*term73;
  const double term129 = term26*(-l[2][0].z()*term115 + l[2][1].z()*term118 + l[2][2].z()*term119 + l[2][3].z()*term116 + term121*term66 + term122*term67) + term31*(-l[1][0].z()*term115 + l[1][1].z()*term118 + l[1][2].z()*term119 + l[1][3].z()*term116 + term121*term70 + term122*term71) + term77*(-l[0][0].z()*term115 + l[0][1].z()*term118 + l[0][2].z()*term119 + l[0][3].z()*term116 + term121*term56 + term122*term57) + term81*(-l[3][0].z()*term115 + l[3][1].z()*term118 + l[3][2].z()*term119 + l[3][3].z()*term116 + term121*term61 + term122*term62);
  const double term130 = 2*term124*(term123*term125 + term126*term127 + term128*term129);
  const double term131 = 1.5*term24;
  const double term132 = 10*frac_y - term131 - 16;
  const double term133 = term132*term29;
  const double term134 = -11.0*frac_y + term131 + 19.5;
  const double term135 = term134*term34;
  const double term136 = 2*frac_y;
  const double term137 = term1*(term136 - 8);
  const double term138 = term137*term14;
  const double term139 = term17*(term136 - 6);
  const double term140 = term139*term21;
  const double term141 = term132*term49;
  const double term142 = term134*term53;
  const double term143 = term137*term39;
  const double term144 = term139*term44;
  const double term145 = term132*term68;
  const double term146 = term134*term72;
  const double term147 = term137*term58;
  const double term148 = term139*term63;
  const double term149 = term124*(term125*(2*term133 + 2*term135 + 2*term138 + 2*term140 - 0.333333333333333*term15 + 0.333333333333333*term22) + term126*(2*term141 + 2*term142 + 2*term143 + 2*term144 - 0.333333333333333*term40 + 0.333333333333333*term45) + term128*(2*term145 + 2*term146 + 2*term147 + 2*term148 - 0.333333333333333*term59 + 0.333333333333333*term64));
  const double term150 = 0.166666666666667*term2;
  const double term151 = 0.166666666666667*term18;
  
  result->direction().x() = term36*term76;
  result->direction().y() = term55*term76;
  result->direction().z() = term74*term76;
  result->origin().x() = term26*term87 + term31*term90 + term77*term80 + term81*term84;
  result->origin().y() = term102*term31 + term26*term99 + term77*term93 + term81*term96;
  result->origin().z() = term105*term77 + term108*term81 + term111*term26 + term114*term31;
  (*dresult_dxy)(0, 0) = term123*term76 - term130*term36;
  (*dresult_dxy)(0, 1) = -term149*term36 + term76*(term133 + term135 + term138 + term140 - 0.166666666666667*term15 + 0.166666666666667*term22);
  (*dresult_dxy)(1, 0) = term127*term76 - term130*term55;
  (*dresult_dxy)(1, 1) = -term149*term55 + term76*(term141 + term142 + term143 + term144 - 0.166666666666667*term40 + 0.166666666666667*term45);
  (*dresult_dxy)(2, 0) = term129*term76 - term130*term74;
  (*dresult_dxy)(2, 1) = -term149*term74 + term76*(term145 + term146 + term147 + term148 - 0.166666666666667*term59 + 0.166666666666667*term64);
  (*dresult_dxy)(3, 0) = term26*(-l[2][0](3)*term115 + l[2][1](3)*term118 + l[2][2](3)*term119 + l[2][3](3)*term116 + term121*term85 + term122*term86) + term31*(-l[1][0](3)*term115 + l[1][1](3)*term118 + l[1][2](3)*term119 + l[1][3](3)*term116 + term121*term88 + term122*term89) + term77*(-l[0][0](3)*term115 + l[0][1](3)*term118 + l[0][2](3)*term119 + l[0][3](3)*term116 + term121*term78 + term122*term79) + term81*(-l[3][0](3)*term115 + l[3][1](3)*term118 + l[3][2](3)*term119 + l[3][3](3)*term116 + term121*term82 + term122*term83);
  (*dresult_dxy)(3, 1) = term132*term87 + term134*term90 + term137*term80 + term139*term84 - term150*term80 + term151*term84;
  (*dresult_dxy)(4, 0) = term26*(-l[2][0](4)*term115 + l[2][1](4)*term118 + l[2][2](4)*term119 + l[2][3](4)*term116 + term121*term97 + term122*term98) + term31*(-l[1][0](4)*term115 + l[1][1](4)*term118 + l[1][2](4)*term119 + l[1][3](4)*term116 + term100*term121 + term101*term122) + term77*(-l[0][0](4)*term115 + l[0][1](4)*term118 + l[0][2](4)*term119 + l[0][3](4)*term116 + term121*term91 + term122*term92) + term81*(-l[3][0](4)*term115 + l[3][1](4)*term118 + l[3][2](4)*term119 + l[3][3](4)*term116 + term121*term94 + term122*term95);
  (*dresult_dxy)(4, 1) = term102*term134 + term132*term99 + term137*term93 + term139*term96 - term150*term93 + term151*term96;
  (*dresult_dxy)(5, 0) = term26*(-l[2][0](5)*term115 + l[2][1](5)*term118 + l[2][2](5)*term119 + l[2][3](5)*term116 + term109*term121 + term110*term122) + term31*(-l[1][0](5)*term115 + l[1][1](5)*term118 + l[1][2](5)*term119 + l[1][3](5)*term116 + term112*term121 + term113*term122) + term77*(-l[0][0](5)*term115 + l[0][1](5)*term118 + l[0][2](5)*term119 + l[0][3](5)*term116 + term103*term121 + term104*term122) + term81*(-l[3][0](5)*term115 + l[3][1](5)*term118 + l[3][2](5)*term119 + l[3][3](5)*term116 + term106*term121 + term107*term122);
  (*dresult_dxy)(5, 1) = term105*term137 - term105*term150 + term108*term139 + term108*term151 + term111*term132 + term114*term134;
}
}  // namespace aslam
