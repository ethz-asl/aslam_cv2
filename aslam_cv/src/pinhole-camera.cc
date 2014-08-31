#include <aslam/cameras/pinhole-camera.h>
// TODO(slynen) Enable commented out PropertyTree support
//#include <sm/PropertyTree.hpp>

namespace aslam {
PinholeCamera::PinholeCamera()
: _intrinsics(kNumOfParams){
  _intrinsics << 0, 0, 0, 0;
  // TODO(PTF) make sure this can work with null distortion
  updateTemporaries();
}

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
//  updateTemporaries();
//}

PinholeCamera::PinholeCamera(double focalLengthCols, double focalLengthRows,
                             double imageCenterCols, double imageCenterRows,
                             uint32_t imageWidth, uint32_t imageHeight,
                             aslam::Distortion::Ptr distortion,
                             const Eigen::VectorXd& distortionParams)
: _intrinsics(kNumOfParams + (distortion? distortion->getParameterSize() : 0)),
  _distortion(distortion) {
  CHECK_NOTNULL(distortion.get());
  CHECK(distortion->distortionParametersValid(vecmap(distortionParams)));
  _intrinsics << focalLengthCols, focalLengthRows, imageCenterCols, imageCenterRows, distortionParams;
  setImageWidth(imageWidth);
  setImageHeight(imageHeight);
  updateTemporaries();
}

PinholeCamera::PinholeCamera(double focalLengthCols, double focalLengthRows,
                             double imageCenterCols, double imageCenterRows,
                             uint32_t imageWidth, uint32_t imageHeight)
: _intrinsics(kNumOfParams) {
  _intrinsics << focalLengthCols, focalLengthRows, imageCenterCols, imageCenterRows;
  setImageWidth(imageWidth);
  setImageHeight(imageHeight);
  updateTemporaries();
}

PinholeCamera::~PinholeCamera() {}

bool PinholeCamera::operator==(const PinholeCamera& other) const {
  bool same = Camera::operator==(other);
  same &= _intrinsics == other._intrinsics;
  same &= static_cast<bool>(_distortion) == static_cast<bool>(other._distortion);
  if (static_cast<bool>(_distortion) && static_cast<bool>(other._distortion)) {
    same &= *_distortion == *other._distortion;
  }
  return same;
}

bool PinholeCamera::project3(
    const Eigen::Matrix<double, 3, 1>& p,
    Eigen::Matrix<double, 2, 1>* outKeypoint) const {
  CHECK_NOTNULL(outKeypoint);
  const double& fu = _intrinsics(0);
  const double& fv = _intrinsics(1);
  const double& cu = _intrinsics(2);
  const double& cv = _intrinsics(3);

  double rz = 1.0 / p[2];
  Eigen::Matrix<double, 2, 1> keypoint;
  keypoint[0] = p[0] * rz;
  keypoint[1] = p[1] * rz;

  CHECK_NOTNULL(_distortion.get());
  _distortion->distort(getDistortionParameters(), keypoint, outKeypoint);

  (*outKeypoint)[0] = fu * (*outKeypoint)[0] + cu;
  (*outKeypoint)[1] = fv * (*outKeypoint)[1] + cv;

  return isVisible(*outKeypoint) && (p[2] > kMinimumDepth);
}

bool PinholeCamera::project3(
    const Eigen::Matrix<double, 3, 1>& p,
    Eigen::Matrix<double, 2, 1>* outKeypoint,
    Eigen::Matrix<double, 2, 3>* outJp) const {
  CHECK_NOTNULL(outKeypoint);
  CHECK_NOTNULL(outJp);
  const double& fu = _intrinsics(0);
  const double& fv = _intrinsics(1);
  const double& cu = _intrinsics(2);
  const double& cv = _intrinsics(3);

  // Jacobian:
  outJp->setZero();

  double rz = 1.0 / p[2];
  double rz2 = rz * rz;
  (*outKeypoint)[0] = p[0] * rz;
  (*outKeypoint)[1] = p[1] * rz;

  Eigen::Matrix<double, 2, Eigen::Dynamic> Jd;
  CHECK_NOTNULL(_distortion.get());
  _distortion->distort(getDistortionParameters(), outKeypoint, &Jd);  // distort and Jacobian wrt. keypoint
  CHECK_GE(Jd.cols(), 2);

  Eigen::Matrix<double, 2, 3>& J = *outJp;
  // Jacobian including distortion
  J(0, 0) = fu * Jd(0, 0) * rz;
  J(0, 1) = fu * Jd(0, 1) * rz;
  J(0, 2) = -fu * (p[0] * Jd(0, 0) + p[1] * Jd(0, 1)) * rz2;
  J(1, 0) = fv * Jd(1, 0) * rz;
  J(1, 1) = fv * Jd(1, 1) * rz;
  J(1, 2) = -fv * (p[0] * Jd(1, 0) + p[1] * Jd(1, 1)) * rz2;

  (*outKeypoint)[0] = fu * (*outKeypoint)[0] + cu;
  (*outKeypoint)[1] = fv * (*outKeypoint)[1] + cv;

  return isVisible(*outKeypoint) && (p[2] > kMinimumDepth);
}

bool PinholeCamera::project4(
    const Eigen::Matrix<double, 4, 1>& ph,
    Eigen::Matrix<double, 2, 1>* outKeypoint) const {
  CHECK_NOTNULL(outKeypoint);
  if (ph[3] < 0)
    return project3(-ph.head<3>(), outKeypoint);
  else
    return project3(ph.head<3>(), outKeypoint);
}

bool PinholeCamera::project4(
    const Eigen::Matrix<double, 4, 1>& ph,
    Eigen::Matrix<double, 2, 1>* outKeypoint,
    Eigen::Matrix<double, 2, 4>* outJp) const {
  CHECK_NOTNULL(outKeypoint);
  CHECK_NOTNULL(outJp);

  Eigen::Matrix<double, 2, 3> J;
  J.setZero();
  bool success = project3(ph.head<3>(), outKeypoint, &J);
  outJp->setZero();
  outJp->topLeftCorner<2, 3>() = J;
  return success;
}

bool PinholeCamera::backProject3(
    const Eigen::Matrix<double, 2, 1>& keypoint,
    Eigen::Matrix<double, 3, 1>* outPoint) const {
  CHECK_NOTNULL(outPoint);
  const double& fu = _intrinsics(0);
  const double& fv = _intrinsics(1);
  const double& cu = _intrinsics(2);
  const double& cv = _intrinsics(3);


  Eigen::Matrix<double, 2, 1> kp = keypoint;
  kp[0] = (kp[0] - cu) / fu;
  kp[1] = (kp[1] - cv) / fv;

  _distortion->undistort(getDistortionParameters(), &kp);  // revert distortion

  (*outPoint)[0] = kp[0];
  (*outPoint)[1] = kp[1];
  (*outPoint)[2] = 1;

  return isVisible(keypoint);
}

bool PinholeCamera::backProject3(
    const Eigen::Matrix<double, 2, 1>& keypoint,
    Eigen::Matrix<double, 3, 1>* outPoint,
    Eigen::Matrix<double, 3, 2>* outJk) const {
  CHECK_NOTNULL(outPoint);
  CHECK_NOTNULL(outJk);
  const double& fu = _intrinsics(0);
  const double& fv = _intrinsics(1);
  const double& cu = _intrinsics(2);
  const double& cv = _intrinsics(3);

  Eigen::Matrix<double, 2, 1> kp = keypoint;

  kp[0] = (kp[0] - cu) / fu;
  kp[1] = (kp[1] - cv) / fv;

  Eigen::Matrix<double, 2, Eigen::Dynamic> Jd;
  _distortion->undistort(getDistortionParameters(), &kp, &Jd);  // revert distortion

  (*outPoint)[0] = kp[0];
  (*outPoint)[1] = kp[1];
  (*outPoint)[2] = 1;

  outJk->setZero();

  (*outJk)(0, 0) = _recip_fu;
  (*outJk)(1, 1) = _recip_fv;

  (*outJk) *= Jd;

  return isVisible(keypoint);

}

bool PinholeCamera::backProject4(
    const Eigen::Matrix<double, 2, 1>& keypoint,
    Eigen::Matrix<double, 4, 1>* outPoint) const {
  CHECK_NOTNULL(outPoint);
  Eigen::Matrix<double, 3, 1> p;
  bool success = backProject3(keypoint, &p);

  (*outPoint) << p, 0.0;
  return success;

}

bool PinholeCamera::backProject4(
    const Eigen::Matrix<double, 2, 1>& keypoint,
    Eigen::Matrix<double, 4, 1>* outPoint,
    Eigen::Matrix<double, 4, 2>* outJk) const {
  CHECK_NOTNULL(outPoint);
  CHECK_NOTNULL(outJk);
  Eigen::Matrix<double, 3, 2> Jk;
  Eigen::Matrix<double, 3, 1> p;
  bool success = backProject3(keypoint, &p, &Jk);

  (*outPoint) << p, 0.0;

  outJk->setZero();
  outJk->topLeftCorner<3, 2>() = Jk;
  return success;
}

bool PinholeCamera::project3IntrinsicsJacobian(
    const Eigen::Matrix<double, 3, 1>& p,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* outJi) const {
  // TODO(PTF) Fix so that it includes the distortion Jacobian
  CHECK_NOTNULL(outJi);
  outJi->resize(Eigen::NoChange, this->getParameterSize());
  Eigen::Matrix<double, 2, Eigen::Dynamic>& J = *outJi;
  J.setZero();

  double rz = 1.0 / p[2];

  Eigen::Matrix<double, 2, 1> kp;
  kp[0] = p[0] * rz;
  kp[1] = p[1] * rz;
  _distortion->distort(getDistortionParameters(), &kp);

  J(0, 0) = kp[0];
  J(0, 2) = 1;

  J(1, 1) = kp[1];
  J(1, 3) = 1;
  return true;
}

bool PinholeCamera::project4IntrinsicsJacobian(
    const Eigen::Matrix<double, 4, 1>& p,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* outJi) const {
  /// TODO(PTF) make it work with the distortion Jacobian
  CHECK_NOTNULL(outJi);
  outJi->resize(Eigen::NoChange, this->getParameterSize());
  Eigen::Matrix<double, 2, Eigen::Dynamic>& J = *outJi;
  J.setZero();

  bool success;
  if (p[3] < 0.0) {
    success = project3IntrinsicsJacobian(-p.head<3>(), outJi);
  } else {
    success = project3IntrinsicsJacobian(p.head<3>(), outJi);
  }
  return success;
}

// \brief creates a random valid keypoint.
Eigen::Matrix<double, 2, 1> PinholeCamera::createRandomKeypoint() const {
  Eigen::Matrix<double, 2, 1> out;
  out.setRandom();
  out(0) = std::abs(out(0)) * imageWidth();
  out(1) = std::abs(out(1)) * imageHeight();
  return out;
}

// \brief creates a random visible point. Negative depth means random between 0 and 100 meters.
Eigen::Matrix<double, 3, 1> PinholeCamera::createRandomVisiblePoint(
    double depth) const {
  Eigen::Matrix<double, 2, 1> y = createRandomKeypoint();
  Eigen::Matrix<double, 3, 1> p;
  backProject3(y, &p);

  if (depth < 0.0) {
    depth = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 100.0;
  }

  p /= p.norm();

  // Muck with the depth. This doesn't change the pointing direction.
  p *= depth;
  return p;
}

bool PinholeCamera::isProjectable3(
    const Eigen::Matrix<double, 3, 1>& p) const {
  Eigen::Matrix<double, 2, 1> k;
  return project3(p, &k);

}

bool PinholeCamera::isProjectable4(
    const Eigen::Matrix<double, 4, 1>& ph) const {
  Eigen::Matrix<double, 2, 1> k;
  return project4(ph, &k);
}

const Eigen::VectorXd& PinholeCamera::getParameters() const {
  return _intrinsics;
}

void PinholeCamera::setParameters(const Eigen::VectorXd& params) {
  CHECK_EQ(getParameterSize(), static_cast<size_t>(params.size()));
  _intrinsics = params;
  updateTemporaries();
}

size_t PinholeCamera::getParameterSize() const {
  return kNumOfParams + (_distortion ? _distortion->getParameterSize() : 0);
}

void PinholeCamera::updateTemporaries() {
  const double& fu = _intrinsics(0);
  const double& fv = _intrinsics(1);

  _recip_fu = 1.0 / fu;
  _recip_fv = 1.0 / fv;
  _fu_over_fv = fu / fv;

}

/// \brief Get a set of border rays
void PinholeCamera::getBorderRays(Eigen::MatrixXd& rays) {
  rays.resize(4, 8);
  Eigen::Matrix<double, 4, 1> ray;
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

bool PinholeCamera::project3Functional(
    const Eigen::VectorXd& /*intrinsics_params*/, const Eigen::Vector3d& /*point*/,
    Eigen::Vector2d* /* out_point */,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* /* out_intrinsics_jacobian */,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* /* out_point_jacobian */) const {
  // TODO(PTF) implement
  CHECK(false) << "Not Implemented";
}

void PinholeCamera::printParameters(std::ostream& out,
                                    const std::string& text) {
  Camera::printParameters(out,text);
  out << "  focal length (cols,rows): "
      << focalLengthCol() << ", " << focalLengthRow() << std::endl;
  out << "  optical center (cols,rows): "
      << opticalCenterCol() << ", " << opticalCenterRow() << std::endl;
}

Eigen::Map<const Eigen::VectorXd> PinholeCamera::getDistortionParameters() const {
  CHECK_NOTNULL(_distortion.get());
  return vecmap(_intrinsics.tail(_distortion->getParameterSize()));
}

namespace detailPinhole {

inline double square(double x) {
  return x * x;
}
inline float square(float x) {
  return x * x;
}
inline double hypot(double a, double b) {
  return sqrt(square(a) + square(b));
}

}  // namespace detail

// TODO(slynen): Reintegrate once we have the calibration targets.
/*
/// \brief initialize the intrinsics based on one view of a gridded calibration target
/// \return true on success
///
/// These functions were developed with the help of Lionel Heng and the excellent camodocal
/// https://github.com/hengli/camodocal
bool PinholeCamera::initializeIntrinsics(const std::vector<GridCalibrationTargetObservation> &observations) {
  SM_DEFINE_EXCEPTION(Exception, std::runtime_error);
  SM_ASSERT_TRUE(Exception, observations.size() != 0, "Need min. one observation");

  if(observations.size()>1)
    SM_DEBUG_STREAM("pinhole camera model only supports one observation for intrinsics initialization! (using first image)");

  GridCalibrationTargetObservation obs = observations[0];

  using detailPinhole::square;
  using detailPinhole::hypot;
  if (!obs.target()) {
    SM_ERROR("The GridCalibrationTargetObservation has no target object");
    return false;
  }

  // First, initialize the image center at the center of the image.
  _cu = (obs.imCols()-1.0) / 2.0;
  _cv = (obs.imRows()-1.0) / 2.0;
  _ru = obs.imCols();
  _rv = obs.imRows();

  _distortion.clear();

  // Grab a reference to the target for easy access.
  const GridCalibrationTargetBase & target = *obs.target();

  /// Initialize some temporaries needed.
  double gamma0 = 0.0;
  double minReprojErr = std::numeric_limits<double>::max();

  // Now we try to find a non-radial line to initialize the focal length
  bool success = false;
  for (size_t r = 0; r < target.rows(); ++r) {
    // Grab all the valid corner points for this checkerboard observation
    cv::Mat P(target.cols(), 4, CV_64F);
    size_t count = 0;
    for (size_t c = 0; c < target.cols(); ++c) {
      Eigen::Vector2d imagePoint;
      Eigen::Vector3d gridPoint;
      if (obs.imageGridPoint(r, c, imagePoint)) {
        double u = imagePoint[0] - _cu;
        double v = imagePoint[1] - _cv;
        P.at<double>(count, 0) = u;
        P.at<double>(count, 1) = v;
        P.at<double>(count, 2) = 0.5;
        P.at<double>(count, 3) = -0.5 * (square(u) + square(v));
        ++count;
      }
    }

    const size_t MIN_CORNERS = 3;
    // MIN_CORNERS is an arbitrary threshold for the number of corners
    if (count > MIN_CORNERS) {
      // Resize P to fit with the count of valid points.
      cv::Mat C;
      cv::SVD::solveZ(P.rowRange(0, count), C);

      double t = square(C.at<double>(0)) + square(C.at<double>(1))
                              + C.at<double>(2) * C.at<double>(3);
      if (t < 0) {
        SM_DEBUG_STREAM("Skipping a bad SVD solution on row " << r);
        continue;
      }

      // check that line image is not radial
      double d = sqrt(1.0 / t);
      double nx = C.at<double>(0) * d;
      double ny = C.at<double>(1) * d;
      if (hypot(nx, ny) > 0.95) {
        SM_DEBUG_STREAM("Skipping a radial line on row " << r);
        continue;
      }

      double nz = sqrt(1.0 - square(nx) - square(ny));
      double gamma = fabs(C.at<double>(2) * d / nz);

      SM_DEBUG_STREAM("Testing a focal length estimate of " << gamma);
      _fu = gamma;
      _fv = gamma;
      updateTemporaries();
      sm::kinematics::Transformation T_target_camera;
      if (!estimateTransformation(obs, T_target_camera)) {
        SM_DEBUG_STREAM(
            "Skipping row " << r
            << " as the transformation estimation failed.");
        continue;
      }

      double reprojErr = 0.0;
      size_t numReprojected = computeReprojectionError(obs, T_target_camera,
                                                       reprojErr);

      if (numReprojected > MIN_CORNERS) {
        double avgReprojErr = reprojErr / numReprojected;

        if (avgReprojErr < minReprojErr) {
          SM_DEBUG_STREAM(
              "Row " << r << " produced the new best estimate: " << avgReprojErr
              << " < " << minReprojErr);
          minReprojErr = avgReprojErr;
          gamma0 = gamma;
          success = true;
        }
      }

    }  // If this observation has enough valid corners
    else {
      SM_DEBUG_STREAM(
          "Skipping row " << r << " because it only had " << count
          << " corners. Minimum: " << MIN_CORNERS);
    }
  }  // For each row in the image.

  _fu = gamma0;
  _fv = gamma0;
  updateTemporaries();
  return success;
}

size_t PinholeCamera::computeReprojectionError(
    const GridCalibrationTargetObservation & obs,
    const sm::kinematics::Transformation & T_target_camera,
    double & outErr) const {
  outErr = 0.0;
  size_t count = 0;
  sm::kinematics::Transformation T_camera_target = T_target_camera.inverse();

  for (size_t i = 0; i < obs.target()->size(); ++i) {
    Eigen::Vector2d y, yhat;
    if (obs.imagePoint(i, y)
        && euclideanToKeypoint(T_camera_target * obs.target()->point(i),
                               yhat)) {
      outErr += (y - yhat).norm();
      ++count;
    }
  }

  return count;
}

/// \brief estimate the transformation of the camera with respect to the calibration target
///        On success out_T_t_c is filled in with the transformation that takes points from
///        the camera frame to the target frame
/// \return true on success
bool PinholeCamera::estimateTransformation(
    const GridCalibrationTargetObservation & obs,
    sm::kinematics::Transformation & out_T_t_c) const {

  std::vector<cv::Point2f> Ms;
  std::vector<cv::Point3f> Ps;

  // Get the observed corners in the image and target frame
  obs.getCornersImageFrame(Ms);
  obs.getCornersTargetFrame(Ps);

  // Convert all target corners to a fakey pinhole view.
  size_t count = 0;
  for (size_t i = 0; i < Ms.size(); ++i) {
    Eigen::Vector3d targetPoint(Ps[i].x, Ps[i].y, Ps[i].z);
    Eigen::Vector2d imagePoint(Ms[i].x, Ms[i].y);
    Eigen::Vector3d backProjection;

    if (keypointToEuclidean(imagePoint, backProjection)
        && backProjection[2] > 0.0) {
      double x = backProjection[0];
      double y = backProjection[1];
      double z = backProjection[2];
      Ps.at(count).x = targetPoint[0];
      Ps.at(count).y = targetPoint[1];
      Ps.at(count).z = targetPoint[2];

      Ms.at(count).x = x / z;
      Ms.at(count).y = y / z;
      ++count;
    } else {
      SM_DEBUG_STREAM(
          "Skipping point " << i << ", point was observed: " << imagePoint
          << ", projection success: "
          << keypointToEuclidean(imagePoint, backProjection)
          << ", in front of camera: " << (backProjection[2] > 0.0)
          << "image point: " << imagePoint.transpose()
          << ", backProjection: " << backProjection.transpose()
          << ", camera params (fu,fv,cu,cv):" << fu() << ", " << fv()
          << ", " << cu() << ", " << cv());
    }
  }

  Ps.resize(count);
  Ms.resize(count);

  std::vector<double> distCoeffs(4, 0.0);

  cv::Mat rvec(3, 1, CV_64F);
  cv::Mat tvec(3, 1, CV_64F);

  if (Ps.size() < 4) {
    SM_DEBUG_STREAM(
        "At least 4 points are needed for calling PnP. Found " << Ps.size());
    return false;
  }

  // Call the OpenCV pnp function.
  SM_DEBUG_STREAM(
      "Calling solvePnP with " << Ps.size() << " world points and " << Ms.size()
      << " image points");
  cv::solvePnP(Ps, Ms, cv::Mat::eye(3, 3, CV_64F), distCoeffs, rvec, tvec);

  // convert the rvec/tvec to a transformation
  cv::Mat C_camera_model = cv::Mat::eye(3, 3, CV_64F);
  Eigen::Matrix4d T_camera_model = Eigen::Matrix4d::Identity();
  cv::Rodrigues(rvec, C_camera_model);
  for (int r = 0; r < 3; ++r) {
    T_camera_model(r, 3) = tvec.at<double>(r, 0);
    for (int c = 0; c < 3; ++c) {
      T_camera_model(r, c) = C_camera_model.at<double>(r, c);
    }
  }

  out_T_t_c.set(T_camera_model.inverse());

  SM_DEBUG_STREAM("solvePnP solution:" << out_T_t_c.T());

  return true;
}
*/



}  // namespace aslam
