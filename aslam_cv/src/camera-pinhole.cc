#include <aslam/cameras/camera-pinhole.h>
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
                             aslam::Distortion::Ptr distortion)
: _intrinsics(kNumOfParams),
  _distortion(distortion) {
  CHECK_NOTNULL(distortion.get());
  _intrinsics << focalLengthCols, focalLengthRows, imageCenterCols, imageCenterRows;
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

bool PinholeCamera::operator==(const Camera& other) const {
  //check if the camera models are the same
  const PinholeCamera* rhs = dynamic_cast<const PinholeCamera*>(&other);

  if (!rhs)
    return false;

  // Verify that the base class members are equal.
  if (!Camera::operator==(other))
    return false;

  // Compare the distortion model (if set).
  if (_distortion && rhs->_distortion) {
    if ( !(*(this->_distortion) == *(rhs->_distortion)) )
      return false;
  } else {
    return false;
  }

  //check intrinsics parameters
  return _intrinsics == rhs->_intrinsics;
}

bool PinholeCamera::project3(const Eigen::Matrix<double, 3, 1>& p,
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
  _distortion->distort(keypoint, outKeypoint);

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

  Eigen::Matrix2d Jd;
  CHECK_NOTNULL(_distortion.get());
  _distortion->distort(outKeypoint, &Jd);  // distort and Jacobian wrt. keypoint
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

  _distortion->undistort(&kp);  // revert distortion

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

  _distortion->undistort(&kp);
  CHECK(false) << "undistort: Jacobian not implemented!";

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
  _distortion->distort(&kp);

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
  return kNumOfParams ;
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

}  // namespace aslam
