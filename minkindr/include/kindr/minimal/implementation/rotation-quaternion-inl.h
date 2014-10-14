#ifndef KINDR_MIN_ROTATION_QUATERNION_INL_H_
#define KINDR_MIN_ROTATION_QUATERNION_INL_H_
#include <kindr/minimal/rotation-quaternion.h>
#include <kindr/minimal/angle-axis.h>
#include <glog/logging.h>
#include <boost/math/special_functions/sinc.hpp>

namespace kindr {
namespace minimal {

/// \brief initialize to identity
template<typename Scalar>
RotationQuaternionTemplate<Scalar>::RotationQuaternionTemplate() :
    q_A_B_(Implementation::Identity()) {
}

/// \brief initialize from real and imaginary components (real first)
template<typename Scalar>
RotationQuaternionTemplate<Scalar>::RotationQuaternionTemplate(
    Scalar w, Scalar x, Scalar y, Scalar z) :
    q_A_B_(w,x,y,z) {
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
}

  
/// \brief initialize from real and imaginary components
template<typename Scalar>
RotationQuaternionTemplate<Scalar>::RotationQuaternionTemplate(
    Scalar real,
    const typename RotationQuaternionTemplate<Scalar>::Vector3& imaginary) :
    q_A_B_(real, imaginary[0], imaginary[1], imaginary[2]){
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0),
             static_cast<Scalar>(1e-4));
}


/// \brief initialize from an Eigen quaternion
template<typename Scalar>
RotationQuaternionTemplate<Scalar>::RotationQuaternionTemplate(
    const Implementation& quaternion) :
    q_A_B_(quaternion){
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0),
             static_cast<Scalar>(1e-4));
}

/// \brief initialize from real and imaginary components (real first)
template<typename Scalar>
RotationQuaternionTemplate<Scalar>::RotationQuaternionTemplate(
    const Vector4& quat) :
    q_A_B_(quat[0], quat[1], quat[2], quat[3])
{
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0),
             static_cast<Scalar>(1e-4));
}

/// \brief initialize from axis-scaled angle vector
template<typename Scalar>
RotationQuaternionTemplate<Scalar>::RotationQuaternionTemplate(
    const Vector3& axis_scale_angle) {
  Scalar half_angle = axis_scale_angle.norm()/static_cast<Scalar>(2.0);
  Scalar half_sinc_angle = (boost::math::sinc_pi(half_angle))/static_cast<Scalar>(2.0);
  q_A_B_ = Implementation(cos(half_angle),
                          half_sinc_angle*axis_scale_angle[0],
                          half_sinc_angle*axis_scale_angle[1],
                          half_sinc_angle*axis_scale_angle[2]);
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
}

/// \brief initialize from a rotation matrix
template<typename Scalar>
RotationQuaternionTemplate<Scalar>::RotationQuaternionTemplate(
    const RotationMatrix& matrix) :
    q_A_B_(matrix) {
  // \todo furgalep check that this was a real rotation matrix
}


template<typename Scalar>
RotationQuaternionTemplate<Scalar>::RotationQuaternionTemplate(
    const AngleAxisTemplate<Scalar>& angleAxis) :
    q_A_B_(angleAxis.toImplementation()){

}


template<typename Scalar>
RotationQuaternionTemplate<Scalar>::~RotationQuaternionTemplate() {

}


/// \brief the real component of the quaternion
template<typename Scalar>
Scalar RotationQuaternionTemplate<Scalar>::w() const {
  return q_A_B_.w();
}

/// \brief the first imaginary component of the quaternion
template<typename Scalar>
Scalar RotationQuaternionTemplate<Scalar>::x() const {
  return q_A_B_.x();
}

/// \brief the second imaginary component of the quaternion
template<typename Scalar>
Scalar RotationQuaternionTemplate<Scalar>::y() const {
  return q_A_B_.y();
}

/// \brief the third imaginary component of the quaternion
template<typename Scalar>
Scalar RotationQuaternionTemplate<Scalar>::z() const {
  return q_A_B_.z();
}

/// \brief assignment operator
template<typename Scalar>
RotationQuaternionTemplate<Scalar>&
RotationQuaternionTemplate<Scalar>::operator=(
    const RotationQuaternionTemplate<Scalar>& rhs) {
  if(this != &rhs) {
    q_A_B_ = rhs.q_A_B_;
  }
  return *this;
}

/// \brief the imaginary components of the quaterion.
template<typename Scalar>
typename RotationQuaternionTemplate<Scalar>::Imaginary
RotationQuaternionTemplate<Scalar>::imaginary() const {
  return Imaginary(q_A_B_.x(),q_A_B_.y(),q_A_B_.z());
}

/// \brief get the components of the quaternion as a vector (real first)
template<typename Scalar>
typename RotationQuaternionTemplate<Scalar>::Vector4
RotationQuaternionTemplate<Scalar>::vector() const {
  return Vector4(q_A_B_.w(), q_A_B_.x(),q_A_B_.y(),q_A_B_.z());
}

/// \brief set the quaternion by its values (real, imaginary)
template<typename Scalar>
void RotationQuaternionTemplate<Scalar>::setValues(Scalar w, Scalar x,
                                                   Scalar y, Scalar z) {
  q_A_B_ = Implementation(w,x,y,z);
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0),
             static_cast<Scalar>(1e-4));
}

/// \brief set the quaternion by its real and imaginary parts
template<typename Scalar>
void RotationQuaternionTemplate<Scalar>::setParts(Scalar real,
                                                  const Imaginary& imag) {
  q_A_B_ = Implementation(real, imag[0], imag[1], imag[2]);
  CHECK_NEAR(squaredNorm(), static_cast<Scalar>(1.0),
             static_cast<Scalar>(1e-4));
}


/// \brief get a copy of the representation that is unique
template<typename Scalar>
RotationQuaternionTemplate<Scalar>
RotationQuaternionTemplate<Scalar>::getUnique() const {
  if(this->w() > 0) {
    return *this;
  } else if (this->w() < 0){
    return RotationQuaternionTemplate<Scalar>(
        -this->w(),-this->x(),-this->y(),-this->z());
  } 
  // w == 0
  if(this->x() > 0) {
    return *this;
  } else if (this->x() < 0){
    return RotationQuaternionTemplate<Scalar>(
        -this->w(),-this->x(),-this->y(),-this->z());
  }
  // x == 0
  if(this->y() > 0) {
    return *this;
  } else if (this->y() < 0){
    return RotationQuaternionTemplate<Scalar>(
        -this->w(),-this->x(),-this->y(),-this->z());
  } 
  // y == 0
  if(this->z() > 0) { // z must be either -1 or 1 in this case
    return *this;
  } else {
    return RotationQuaternionTemplate<Scalar>(
        -this->w(),-this->x(),-this->y(),-this->z());
  }
}

/// \brief set the quaternion to its unique representation
template<typename Scalar>
RotationQuaternionTemplate<Scalar>&
RotationQuaternionTemplate<Scalar>::setUnique() {
  *this = getUnique();
  return *this;
}

/// \brief set the quaternion to identity
template<typename Scalar>
RotationQuaternionTemplate<Scalar>&
RotationQuaternionTemplate<Scalar>::setIdentity() {
  q_A_B_.setIdentity();
  return *this;
}

/// \brief get a copy of the quaternion inverted.
template<typename Scalar>
RotationQuaternionTemplate<Scalar>
RotationQuaternionTemplate<Scalar>::inverted() const {
  return conjugated();
}

/// \brief get a copy of the conjugate of the quaternion.
template<typename Scalar>
RotationQuaternionTemplate<Scalar>
RotationQuaternionTemplate<Scalar>::conjugated() const {
  // Own implementation since Eigen::conjugate does not use the correct
  // scalar type for the greater than zero comparison.
  return RotationQuaternionTemplate(
      Implementation(q_A_B_.w(),-q_A_B_.x(),-q_A_B_.y(),-q_A_B_.z()));
}


/// \brief rotate a vector, v
template<typename Scalar>
typename RotationQuaternionTemplate<Scalar>::Vector3
RotationQuaternionTemplate<Scalar>::rotate(
    const typename RotationQuaternionTemplate<Scalar>::Vector3& v) const {
  return q_A_B_*v;
}


/// \brief rotate a vector, v
template<typename Scalar>
typename RotationQuaternionTemplate<Scalar>::Vector4
RotationQuaternionTemplate<Scalar>::rotate4(
    const typename RotationQuaternionTemplate<Scalar>::Vector4& v) const {
  typename RotationQuaternionTemplate<Scalar>::Vector4 vprime;
  vprime[3] = v[3];
  vprime.template head<3>() = q_A_B_*v.template head<3>();
  return vprime;
}


/// \brief rotate a vector, v
template<typename Scalar>
typename RotationQuaternionTemplate<Scalar>::Vector3
RotationQuaternionTemplate<Scalar>::inverseRotate(
    const typename RotationQuaternionTemplate<Scalar>::Vector3& v) const {
  return q_A_B_.inverse()*v;
}


/// \brief rotate a vector, v
template<typename Scalar>
typename RotationQuaternionTemplate<Scalar>::Vector4
RotationQuaternionTemplate<Scalar>::inverseRotate4(
    const typename RotationQuaternionTemplate<Scalar>::Vector4& v) const {
  typename RotationQuaternionTemplate<Scalar>::Vector4 vprime;
  vprime[3] = v[3];
  vprime.template head<3>() = q_A_B_.inverse()*v.template head<3>();
  return vprime;
}


/// \brief cast to the implementation type
template<typename Scalar>
typename  RotationQuaternionTemplate<Scalar>::Implementation&
RotationQuaternionTemplate<Scalar>::toImplementation() {
  return q_A_B_;
}

/// \brief cast to the implementation type
template<typename Scalar>
const typename RotationQuaternionTemplate<Scalar>::Implementation&
RotationQuaternionTemplate<Scalar>::toImplementation() const {
  return q_A_B_;
}

/// \brief get the norm of the quaternion
template<typename Scalar>
Scalar RotationQuaternionTemplate<Scalar>::norm() const {
  return q_A_B_.norm();
}

/// \brief get the squared norm of the quaternion
template<typename Scalar>
Scalar RotationQuaternionTemplate<Scalar>::squaredNorm() const {
  return q_A_B_.squaredNorm();
}

/// \brief enforce the unit length constraint
template<typename Scalar>
RotationQuaternionTemplate<Scalar>&
RotationQuaternionTemplate<Scalar>::normalize() {
  q_A_B_.normalize();
  return *this;
}

/// \brief compose two quaternions
template<typename Scalar>
RotationQuaternionTemplate<Scalar>
RotationQuaternionTemplate<Scalar>::operator*(
    const RotationQuaternionTemplate<Scalar>& rhs) const {
  return RotationQuaternionTemplate<Scalar>(q_A_B_ * rhs.q_A_B_);
}

template<typename Scalar>
std::ostream& operator<<(std::ostream& out,
                         const RotationQuaternionTemplate<Scalar>& rhs) {
  out << rhs.vector();
  return out;
}

/// \brief get the rotation matrix
template<typename Scalar>
typename RotationQuaternionTemplate<Scalar>::RotationMatrix
RotationQuaternionTemplate<Scalar>::getRotationMatrix() const {
  return q_A_B_.matrix();
}

/// \brief get the angle between this and the other quaternion
template<typename Scalar>
Scalar RotationQuaternionTemplate<Scalar>::getDisparityAngle(
    const RotationQuaternionTemplate<Scalar>& rhs) const{
  return AngleAxis( rhs * this->inverted() ).getUnique().angle();
}

} // namespace minimal
} // namespace kindr
#endif  // KINDR_MIN_ROTATION_QUATERNION_INL_H_
