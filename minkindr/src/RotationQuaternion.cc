#include <kindr/minimal/RotationQuaternion.h>
#include <kindr/minimal/AngleAxis.h>
#include <glog/logging.h>

namespace kindr {
namespace minimal {

/// \brief initialize to identity
RotationQuaternion::RotationQuaternion() : 
    q_A_B_(Implementation::Identity()) {
}

/// \brief initialize from real and imaginary components (real first)
RotationQuaternion::RotationQuaternion(Scalar w, Scalar x, Scalar y, Scalar z) :
    q_A_B_(w,x,y,z) {
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
}

  
/// \brief initialize from real and imaginary components
RotationQuaternion::RotationQuaternion(Scalar real, const Eigen::Vector3d& imaginary) :
    q_A_B_(real, imaginary[0], imaginary[1], imaginary[2]){
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
}


/// \brief initialize from an Eigen quaternion
RotationQuaternion::RotationQuaternion(const Implementation& quaternion) :
    q_A_B_(quaternion){
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
}

/// \brief initialize from real and imaginary components (real first)
RotationQuaternion::RotationQuaternion(const Vector4& quat) :
    q_A_B_(quat[0], quat[1], quat[2], quat[3])
{
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
}

/// \brief initialize from a rotation matrix
RotationQuaternion::RotationQuaternion(const RotationMatrix& matrix) :
    q_A_B_(matrix) {
  // \todo furgalep check that this was a real rotation matrix
}


RotationQuaternion::RotationQuaternion(const AngleAxis& angleAxis) :
    q_A_B_(angleAxis.toImplementation()){

}

  
RotationQuaternion::~RotationQuaternion() {

}


/// \brief the real component of the quaternion
RotationQuaternion::Scalar RotationQuaternion::w() const {
  return q_A_B_.w();
}

/// \brief the first imaginary component of the quaternion
RotationQuaternion::Scalar RotationQuaternion::x() const {
  return q_A_B_.x();
}

/// \brief the second imaginary component of the quaternion
RotationQuaternion::Scalar RotationQuaternion::y() const {
  return q_A_B_.y();
}

/// \brief the third imaginary component of the quaternion
RotationQuaternion::Scalar RotationQuaternion::z() const {
  return q_A_B_.z();
}

/// \brief assignment operator
RotationQuaternion& RotationQuaternion::operator=(const RotationQuaternion& rhs) {
  if(this != &rhs) {
    q_A_B_ = rhs.q_A_B_;
  }
  return *this;
}

/// \brief the imaginary components of the quaterion.
RotationQuaternion::Imaginary RotationQuaternion::imaginary() const {
  return Imaginary(q_A_B_.x(),q_A_B_.y(),q_A_B_.z());
}

/// \brief get the components of the quaternion as a vector (real first)
RotationQuaternion::Vector4 RotationQuaternion::vector() const {
  return Vector4(q_A_B_.w(), q_A_B_.x(),q_A_B_.y(),q_A_B_.z());
}

/// \brief set the quaternion by its values (real, imaginary)
void RotationQuaternion::setValues(Scalar w, Scalar x, Scalar y, Scalar z) {
  q_A_B_ = Implementation(w,x,y,z);
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
}

/// \brief set the quaternion by its real and imaginary parts
void RotationQuaternion::setParts(Scalar real, const Imaginary& imag) {
  q_A_B_ = Implementation(real, imag[0], imag[1], imag[2]);
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
}


/// \brief get a copy of the representation that is unique
RotationQuaternion RotationQuaternion::getUnique() const {
  if(this->w() > 0) {
    return *this;
  } else if (this->w() < 0){
    return RotationQuaternion(-this->w(),-this->x(),-this->y(),-this->z());
  } 
  // w == 0
  if(this->x() > 0) {
    return *this;
  } else if (this->x() < 0){
    return RotationQuaternion(-this->w(),-this->x(),-this->y(),-this->z());
  }
  // x == 0
  if(this->y() > 0) {
    return *this;
  } else if (this->y() < 0){
    return RotationQuaternion(-this->w(),-this->x(),-this->y(),-this->z());
  } 
  // y == 0
  if(this->z() > 0) { // z must be either -1 or 1 in this case
    return *this;
  } else {
    return RotationQuaternion(-this->w(),-this->x(),-this->y(),-this->z());
  }
}

/// \brief set the quaternion to its unique representation
RotationQuaternion& RotationQuaternion::setUnique() {
  *this = getUnique();
  return *this;
}

/// \brief set the quaternion to identity
RotationQuaternion& RotationQuaternion::setIdentity() {
  q_A_B_.setIdentity();
  return *this;
}

/// \brief get a copy of the quaternion inverted.
RotationQuaternion RotationQuaternion::inverted() const {
  return conjugated();
}

/// \brief get a copy of the conjugate of the quaternion.
RotationQuaternion RotationQuaternion::conjugated() const {
  return RotationQuaternion(q_A_B_.inverse());
}


/// \brief rotate a vector, v
Eigen::Vector3d RotationQuaternion::rotate(const Eigen::Vector3d& v) const {
  return q_A_B_*v;
}


/// \brief rotate a vector, v
Eigen::Vector4d RotationQuaternion::rotate4(const Eigen::Vector4d& v) const {
  Eigen::Vector4d vprime;
  vprime[3] = v[3];
  vprime.head<3>() = q_A_B_*v.head<3>();
  return vprime;
}


/// \brief rotate a vector, v
Eigen::Vector3d RotationQuaternion::inverseRotate(const Eigen::Vector3d& v) const {
  return q_A_B_.inverse()*v;
}


/// \brief rotate a vector, v
Eigen::Vector4d RotationQuaternion::inverseRotate4(const Eigen::Vector4d& v) const {
  Eigen::Vector4d vprime;
  vprime[3] = v[3];
  vprime.head<3>() = q_A_B_.inverse()*v.head<3>();
  return vprime;
}


/// \brief cast to the implementation type
RotationQuaternion::Implementation& RotationQuaternion::toImplementation() {
  return q_A_B_;
}

/// \brief cast to the implementation type
const RotationQuaternion::Implementation& RotationQuaternion::toImplementation() const {
  return q_A_B_;
}

/// \brief get the norm of the quaternion
RotationQuaternion::Scalar RotationQuaternion::norm() const {
  return q_A_B_.norm();
}

/// \brief get the squared norm of the quaternion
RotationQuaternion::Scalar RotationQuaternion::squaredNorm() const {
  return q_A_B_.squaredNorm();
}

/// \brief enforce the unit length constraint
RotationQuaternion& RotationQuaternion::normalize() {
  q_A_B_.normalize();
  return *this;
}

/// \brief compose two quaternions
RotationQuaternion RotationQuaternion::operator*(const RotationQuaternion& rhs) const {
  return RotationQuaternion(q_A_B_ * rhs.q_A_B_);
}

std::ostream& operator<<(std::ostream& out, const RotationQuaternion& rhs) {
  out << rhs.vector();
  return out;
}

/// \brief get the rotation matrix
RotationQuaternion::RotationMatrix RotationQuaternion::getRotationMatrix() const {
  return q_A_B_.matrix();
}

/// \brief get the angle between this and the other quaternion
RotationQuaternion::Scalar RotationQuaternion::getDisparityAngle(const RotationQuaternion& rhs) const{
  return AngleAxis( rhs * this->inverted() ).getUnique().angle();
}

} // namespace minimal
} // namespace kindr
