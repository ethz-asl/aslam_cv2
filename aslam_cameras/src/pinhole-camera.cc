#include <aslam/cameras/pinhole-camera.h>
#include <sm/PropertyTree.hpp>

namespace aslam {
PinholeCamera::PinholeCamera()
: _fu(0.0),
  _fv(0.0),
  _cu(0.0),
  _cv(0.0),
  _ru(0),
  _rv(0) {
  updateTemporaries();
}

//PinholeCamera::PinholeCamera(
//    const sm::PropertyTree & config)
//: Camera(config) {
//  _fu = config.getDouble("fu");
//  _fv = config.getDouble("fv");
//  _cu = config.getDouble("cu");
//  _cv = config.getDouble("cv");
//  _ru = config.getInt("ru");
//  _rv = config.getInt("rv");
//
//  //TODO(slynen): Load and instantiate correct distortion here.
//  // distortion.(config, "distortion")
//  CHECK(false) << "Loading of distortion from property tree not implemented.";
//
//  updateTemporaries();
//}

PinholeCamera::PinholeCamera(double focalLengthU,
                             double focalLengthV,
                             double imageCenterU,
                             double imageCenterV,
                             int resolutionU,
                             int resolutionV,
                             aslam::Distortion::Ptr distortion)
: _fu(focalLengthU),
  _fv(focalLengthV),
  _cu(imageCenterU),
  _cv(imageCenterV),
  _ru(resolutionU),
  _rv(resolutionV),
  _distortion(distortion) {
  updateTemporaries();
}

PinholeCamera::PinholeCamera(double focalLengthU,
                             double focalLengthV,
                             double imageCenterU,
                             double imageCenterV,
                             int resolutionU,
                             int resolutionV)
: _fu(focalLengthU),
  _fv(focalLengthV),
  _cu(imageCenterU),
  _cv(imageCenterV),
  _ru(resolutionU),
  _rv(resolutionV) {
  updateTemporaries();
}

PinholeCamera::~PinholeCamera() {}

bool PinholeCamera::operator==(const PinholeCamera& other) const {
  bool same = true;
  same &= _fu == other._fu;
  same &= _fv == other._fv;
  same &= _cu == other._cu;
  same &= _cv == other._cv;
  same &= static_cast<bool>(_distortion) == static_cast<bool>(other._distortion);
  if (static_cast<bool>(_distortion) && static_cast<bool>(other._distortion)) {
    same &= *_distortion == *other._distortion;
  }
  return same;
}

bool PinholeCamera::euclideanToKeypoint(
    const Eigen::Matrix<double, 3, 1>& p,
    Eigen::Matrix<double, 2, 1>* outKeypoint) const {
  CHECK_NOTNULL(outKeypoint);

  double rz = 1.0 / p[2];
  Eigen::Matrix<double, 2, 1> keypoint;
  keypoint[0] = p[0] * rz;
  keypoint[1] = p[1] * rz;

  CHECK_NOTNULL(_distortion.get());
  _distortion->distort(keypoint, outKeypoint);

  (*outKeypoint)[0] = _fu * (*outKeypoint)[0] + _cu;
  (*outKeypoint)[1] = _fv * (*outKeypoint)[1] + _cv;

  return isValid(*outKeypoint) && p[2] > 0;
}

bool PinholeCamera::euclideanToKeypoint(
    const Eigen::Matrix<double, 3, 1>& p,
    Eigen::Matrix<double, 2, 1>* outKeypoint,
    Eigen::Matrix<double, 2, 3>* outJp) const {
  CHECK_NOTNULL(outKeypoint);
  CHECK_NOTNULL(outJp);

  // Jacobian:
  outJp->setZero();

  double rz = 1.0 / p[2];
  double rz2 = rz * rz;
  Eigen::Matrix<double, 2, 1> keypoint;
  keypoint[0] = p[0] * rz;
  keypoint[1] = p[1] * rz;

  Eigen::Matrix<double, 2, Eigen::Dynamic> Jd;
  CHECK_NOTNULL(_distortion.get());
  _distortion->distort(&keypoint, &Jd);  // distort and Jacobian wrt. keypoint
  CHECK_GE(Jd.cols(), 2);

  Eigen::Matrix<double, 2, 3>& J = *outJp;
  // Jacobian including distortion
  J(0, 0) = _fu * Jd(0, 0) * rz;
  J(0, 1) = _fu * Jd(0, 1) * rz;
  J(0, 2) = -_fu * (p[0] * Jd(0, 0) + p[1] * Jd(0, 1)) * rz2;
  J(1, 0) = _fv * Jd(1, 0) * rz;
  J(1, 1) = _fv * Jd(1, 1) * rz;
  J(1, 2) = -_fv * (p[0] * Jd(1, 0) + p[1] * Jd(1, 1)) * rz2;

  (*outKeypoint)[0] = _fu * (*outKeypoint)[0] + _cu;
  (*outKeypoint)[1] = _fv * (*outKeypoint)[1] + _cv;

  return isValid(*outKeypoint) && p[2] > 0;

}

bool PinholeCamera::homogeneousToKeypoint(
    const Eigen::Matrix<double, 4, 1>& ph,
    Eigen::Matrix<double, 2, 1>* outKeypoint) const {
  CHECK_NOTNULL(outKeypoint);
  if (ph[3] < 0)
    return euclideanToKeypoint(-ph.head<3>(), outKeypoint);
  else
    return euclideanToKeypoint(ph.head<3>(), outKeypoint);
}

bool PinholeCamera::homogeneousToKeypoint(
    const Eigen::Matrix<double, 4, 1>& ph,
    Eigen::Matrix<double, 2, 1>* outKeypoint,
    Eigen::Matrix<double, 2, 4>* outJp) const {
  CHECK_NOTNULL(outKeypoint);
  CHECK_NOTNULL(outJp);

  Eigen::Matrix<double, 2, 3> J;
  J.setZero();
  bool success = euclideanToKeypoint(ph.head<3>(), outKeypoint, &J);
  outJp->setZero();
  outJp->topLeftCorner<2, 3>() = J;
  return success;
}

bool PinholeCamera::keypointToEuclidean(
    const Eigen::Matrix<double, 2, 1>& keypoint,
    Eigen::Matrix<double, 3, 1>* outPoint) const {
  CHECK_NOTNULL(outPoint);

  Eigen::Matrix<double, 2, 1> kp = keypoint;
  kp[0] = (kp[0] - _cu) / _fu;
  kp[1] = (kp[1] - _cv) / _fv;

  _distortion->undistort(&kp);  // revert distortion

  (*outPoint)[0] = kp[0];
  (*outPoint)[1] = kp[1];
  (*outPoint)[2] = 1;

  return isValid(keypoint);
}

bool PinholeCamera::keypointToEuclidean(
    const Eigen::Matrix<double, 2, 1>& keypoint,
    Eigen::Matrix<double, 3, 1>* outPoint,
    Eigen::Matrix<double, 3, 2>* outJk) const {
  CHECK_NOTNULL(outPoint);
  CHECK_NOTNULL(outJk);
  Eigen::Matrix<double, 2, 1> kp = keypoint;

  kp[0] = (kp[0] - _cu) / _fu;
  kp[1] = (kp[1] - _cv) / _fv;

  Eigen::Matrix<double, 2, Eigen::Dynamic> Jd;
  _distortion->undistort(&kp, &Jd);  // revert distortion

  (*outPoint)[0] = kp[0];
  (*outPoint)[1] = kp[1];
  (*outPoint)[2] = 1;

  outJk->setZero();

  (*outJk)(0, 0) = _recip_fu;
  (*outJk)(1, 1) = _recip_fv;

  (*outJk) *= Jd;

  return isValid(keypoint);

}

bool PinholeCamera::keypointToHomogeneous(
    const Eigen::Matrix<double, 2, 1>& keypoint,
    Eigen::Matrix<double, 4, 1>* outPoint) const {
  CHECK_NOTNULL(outPoint);
  Eigen::Matrix<double, 3, 1> p;
  bool success = keypointToEuclidean(keypoint, &p);

  (*outPoint) << p, 0.0;
  return success;

}

bool PinholeCamera::keypointToHomogeneous(
    const Eigen::Matrix<double, 2, 1>& keypoint,
    Eigen::Matrix<double, 4, 1>* outPoint,
    Eigen::Matrix<double, 4, 2>* outJk) const {
  CHECK_NOTNULL(outPoint);
  CHECK_NOTNULL(outJk);
  Eigen::Matrix<double, 3, 2> Jk;
  Eigen::Matrix<double, 3, 1> p;
  bool success = keypointToEuclidean(keypoint, &p, &Jk);

  (*outPoint) << p, 0.0;

  outJk->setZero();
  outJk->topLeftCorner<3, 2>() = Jk;
  return success;
}

bool PinholeCamera::euclideanToKeypointIntrinsicsJacobian(
    const Eigen::Matrix<double, 3, 1>& p,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* outJi) const {
  CHECK_NOTNULL(outJi);
  outJi->resize(Eigen::NoChange, 4);
  Eigen::Matrix<double, 2, Eigen::Dynamic>& J = *outJi;
  J.setZero();

  double rz = 1.0 / p[2];

  Eigen::Matrix<double, 2, 1> kp;
  kp[0] = p[0] * rz;
  kp[1] = p[1] * rz;
  _distortion->distort(&kp);

  J(0, 0) = kp[0];
  J(0, 2) = 1;

  J(1, 1) = kp[1];
  J(1, 3) = 1;
  return true;
}

bool PinholeCamera::euclideanToKeypointDistortionJacobian(
    const Eigen::Matrix<double, 3, 1>& p,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* outJd) const {
  CHECK_NOTNULL(outJd);
  outJd->resize(Eigen::NoChange, 4);
  Eigen::Matrix<double, 2, Eigen::Dynamic>& J = *outJd;
  J.setZero();

  double rz = 1.0 / p[2];
  Eigen::Matrix<double, 2, 1> kp;
  kp[0] = p[0] * rz;
  kp[1] = p[1] * rz;

  _distortion->distortParameterJacobian(&kp, &J);

  J.resize(Eigen::NoChange, _distortion->minimalDimensions());
  J.row(0) *= _fu;
  J.row(1) *= _fv;
  return true;
}

bool PinholeCamera::homogeneousToKeypointIntrinsicsJacobian(
    const Eigen::Matrix<double, 4, 1>& p,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* outJi) const {
  CHECK_NOTNULL(outJi);
  outJi->resize(Eigen::NoChange, 4);
  Eigen::Matrix<double, 2, Eigen::Dynamic>& J = *outJi;
  J.setZero();

  bool success;
  if (p[3] < 0.0) {
    success = euclideanToKeypointIntrinsicsJacobian(-p.head<3>(), outJi);
  } else {
    success = euclideanToKeypointIntrinsicsJacobian(p.head<3>(), outJi);
  }
  return success;
}

bool PinholeCamera::homogeneousToKeypointDistortionJacobian(
    const Eigen::Matrix<double, 4, 1>& p,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* outJd) const {
  CHECK_NOTNULL(outJd);
  outJd->resize(Eigen::NoChange, 4);
  Eigen::Matrix<double, 2, Eigen::Dynamic>& J = *outJd;
  J.setZero();

  bool success;
  if (p[3] < 0.0) {
    success = euclideanToKeypointDistortionJacobian(-p.head<3>(), outJd);
  } else {
    success = euclideanToKeypointDistortionJacobian(p.head<3>(), outJd);
  }
  return success;
}

// \brief creates a random valid keypoint.
Eigen::Matrix<double, 2, 1> PinholeCamera::createRandomKeypoint() const {
  Eigen::Matrix<double, 2, 1> out;
  out.setRandom();
  out(0) = std::abs(out(0)) * _ru;
  out(1) = std::abs(out(1)) * _rv;
  return out;
}

// \brief creates a random visible point. Negative depth means random between 0 and 100 meters.
Eigen::Matrix<double, 3, 1> PinholeCamera::createRandomVisiblePoint(
    double depth) const {
  Eigen::Matrix<double, 2, 1> y = createRandomKeypoint();
  Eigen::Matrix<double, 3, 1> p;
  keypointToEuclidean(y, &p);

  if (depth < 0.0) {
    depth = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 100.0;
  }

  p /= p.norm();

  // Muck with the depth. This doesn't change the pointing direction.
  p *= depth;
  return p;
}

bool PinholeCamera::isValid(
    const Eigen::Matrix<double, 2, 1>& keypoint) const {
  return keypoint[0] >= 0 && keypoint[1] >= 0 && keypoint[0] < (double) _ru
      && keypoint[1] < (double) _rv;

}

bool PinholeCamera::isEuclideanVisible(
    const Eigen::Matrix<double, 3, 1>& p) const {
  Eigen::Matrix<double, 2, 1> k;
  return euclideanToKeypoint(p, &k);

}

bool PinholeCamera::isHomogeneousVisible(
    const Eigen::Matrix<double, 4, 1>& ph) const {
  Eigen::Matrix<double, 2, 1> k;
  return homogeneousToKeypoint(ph, &k);
}

void PinholeCamera::update(const double* v) {
  _fu += v[0];
  _fv += v[1];
  _cu += v[2];
  _cv += v[3];
  _recip_fu = 1.0 / _fu;
  _recip_fv = 1.0 / _fv;
  _fu_over_fv = _fu / _fv;
}

int PinholeCamera::minimalDimensions() const {
  return IntrinsicsDimension;
}

void PinholeCamera::getParameters(Eigen::MatrixXd & P) const {
  P.resize(4, 1);
  P(0, 0) = _fu;
  P(1, 0) = _fv;
  P(2, 0) = _cu;
  P(3, 0) = _cv;
}

void PinholeCamera::setParameters(const Eigen::MatrixXd & P) {
  _fu = P(0, 0);
  _fv = P(1, 0);
  _cu = P(2, 0);
  _cv = P(3, 0);
  updateTemporaries();
}

Eigen::Vector2i PinholeCamera::parameterSize() const {
  return Eigen::Vector2i(IntrinsicsDimension, 1);
}

static constexpr int PinholeCamera::parameterCount() const {
  return IntrinsicsDimension;
}

void PinholeCamera::updateTemporaries() {
  _recip_fu = 1.0 / _fu;
  _recip_fv = 1.0 / _fv;
  _fu_over_fv = _fu / _fv;

}

void PinholeCamera::resizeIntrinsics(double scale) {
  _fu *= scale;
  _fv *= scale;
  _cu *= scale;
  _cv *= scale;
  _ru = _ru * scale;
  _rv = _rv * scale;

  updateTemporaries();
}

/// \brief Get a set of border rays
void PinholeCamera::getBorderRays(Eigen::MatrixXd& rays) {
  rays.resize(4, 8);
  Eigen::Matrix<double, 4, 1> ray;
  keypointToHomogeneous(Eigen::Vector2d(0.0, 0.0), &ray);
  rays.col(0) = ray;
  keypointToHomogeneous(Eigen::Vector2d(0.0, _rv * 0.5), &ray);
  rays.col(1) = ray;
  keypointToHomogeneous(Eigen::Vector2d(0.0, _rv - 1.0), &ray);
  rays.col(2) = ray;
  keypointToHomogeneous(Eigen::Vector2d(_ru - 1.0, 0.0), &ray);
  rays.col(3) = ray;
  keypointToHomogeneous(Eigen::Vector2d(_ru - 1.0, _rv * 0.5), &ray);
  rays.col(4) = ray;
  keypointToHomogeneous(Eigen::Vector2d(_ru - 1.0, _rv - 1.0), &ray);
  rays.col(5) = ray;
  keypointToHomogeneous(Eigen::Vector2d(_ru * 0.5, 0.0), &ray);
  rays.col(6) = ray;
  keypointToHomogeneous(Eigen::Vector2d(_ru * 0.5, _rv - 1.0), &ray);
  rays.col(7) = ray;
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
