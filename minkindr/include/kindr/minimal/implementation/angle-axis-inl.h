#ifndef KINDR_MIN_ROTATION_ANGLE_AXIS_INL_H_
#define KINDR_MIN_ROTATION_ANGLE_AXIS_INL_H_
#include <kindr/minimal/angle-axis.h>
#include <kindr/minimal/rotation-quaternion.h>
#include <glog/logging.h>

namespace kindr {
namespace minimal {

/// \brief initialize to identity
template<typename Scalar>
AngleAxisTemplate<Scalar>::AngleAxisTemplate() :
    C_A_B_(Implementation::Identity()) {
}

/// \brief initialize from real and imaginary components (real first)
template<typename Scalar>
AngleAxisTemplate<Scalar>::AngleAxisTemplate(
    Scalar w, Scalar x, Scalar y, Scalar z) : C_A_B_(w, Vector3(x,y,z)) {
  CHECK_NEAR(Vector3(x,y,z).squaredNorm(), static_cast<Scalar>(1.0),
             static_cast<Scalar>(1e-4));
}
 
/// \brief initialize from real and imaginary components
template<typename Scalar>
AngleAxisTemplate<Scalar>::AngleAxisTemplate(
    Scalar angle, const typename AngleAxisTemplate<Scalar>::Vector3& axis) :
    C_A_B_(angle, axis){
    CHECK_NEAR(axis.squaredNorm(), static_cast<Scalar>(1.0),
               static_cast<Scalar>(1e-4));
}

/// \brief initialize from an Eigen angleAxis
template<typename Scalar>
AngleAxisTemplate<Scalar>::AngleAxisTemplate(const Implementation& angleAxis) :
    C_A_B_(angleAxis) {
}

/// \brief initialize from a rotation matrix
template<typename Scalar>
AngleAxisTemplate<Scalar>::AngleAxisTemplate(const RotationMatrix& matrix) :
    C_A_B_(matrix) {
  // \todo furgalep Check that the matrix was good...
}


template<typename Scalar>
AngleAxisTemplate<Scalar>::AngleAxisTemplate(
    const RotationQuaternionTemplate<Scalar>& quat) :
    C_A_B_(quat.toImplementation()) {
}

template<typename Scalar>
AngleAxisTemplate<Scalar>::~AngleAxisTemplate() { }

/// \brief assignment operator
template<typename Scalar>
AngleAxisTemplate<Scalar>& AngleAxisTemplate<Scalar>::operator=(
    const AngleAxisTemplate<Scalar>& rhs) {
  if(this != &rhs) {
    C_A_B_ = rhs.C_A_B_;
  }
  return *this;
}

  /// \brief Returns the rotation angle.
template<typename Scalar>
Scalar AngleAxisTemplate<Scalar>::angle() const{
  return C_A_B_.angle();
}

/// \brief Sets the rotation angle.
template<typename Scalar>
void AngleAxisTemplate<Scalar>::setAngle(Scalar angle){
  C_A_B_.angle() = angle;
}

/// \brief Returns the rotation axis.
template<typename Scalar>
const typename AngleAxisTemplate<Scalar>::Vector3&
AngleAxisTemplate<Scalar>::axis() const{
  return C_A_B_.axis();
}

/// \brief Sets the rotation axis.
template<typename Scalar>
void AngleAxisTemplate<Scalar>::setAxis(const Vector3& axis){
  CHECK_NEAR(axis.squaredNorm(), static_cast<Scalar>(1.0),
             static_cast<Scalar>(1e-4));
  C_A_B_.axis() = axis;
}

/// \brief Sets the rotation axis.
template<typename Scalar>
void AngleAxisTemplate<Scalar>::setAxis(Scalar v1, Scalar v2, Scalar v3){
  C_A_B_.axis() = Vector3(v1,v2,v3);
  CHECK_NEAR(C_A_B_.axis().squaredNorm(), static_cast<Scalar>(1.0),
             static_cast<Scalar>(1e-4));
}

/// \brief get the components of the angle/axis as a vector (real first)
template<typename Scalar>
typename AngleAxisTemplate<Scalar>::Vector4
AngleAxisTemplate<Scalar>::vector() const{
    Vector4 vector;
    vector(0) = angle();
    vector.template tail<3>() = C_A_B_.axis();
    return vector;
}

/// \brief get a copy of the representation that is unique
template<typename Scalar>
AngleAxisTemplate<Scalar> AngleAxisTemplate<Scalar>::getUnique() const {
  // first wraps angle into [-pi,pi)
  AngleAxisTemplate aa(fmod(angle()+M_PI, 2*M_PI)-M_PI, C_A_B_.axis());
    if(aa.angle() > 0)  {
      return aa;
    } else if(aa.angle() < 0) {
      if(aa.angle() != -M_PI) {
        return AngleAxisTemplate(-aa.angle(), -aa.axis());
      } else { // angle == -pi, so axis must be viewed further, because -pi,axis
               // does the same as -pi,-axis.
        if(aa.axis()[0] < 0) {
          return AngleAxisTemplate(-aa.angle(), -aa.axis());
        } else if(aa.axis()[0] > 0) {
          return AngleAxisTemplate(-aa.angle(), aa.axis());
        } else { // v1 == 0
          if(aa.axis()[1] < 0) {
            return AngleAxisTemplate(-aa.angle(), -aa.axis());
          } else if(aa.axis()[1] > 0) {
            return AngleAxisTemplate(-aa.angle(), aa.axis());
          } else { // v2 == 0
            if(aa.axis()[2] < 0) { // v3 must be -1 or 1
              return AngleAxisTemplate(-aa.angle(), -aa.axis());
            } else  {
              return AngleAxisTemplate(-aa.angle(), aa.axis());
            }
          }
        }
      }
    } else { // angle == 0
      return AngleAxisTemplate();
    }
}

/// \brief set the angle/axis to its unique representation
template<typename Scalar>
AngleAxisTemplate<Scalar>& AngleAxisTemplate<Scalar>::setUnique() {
  *this = getUnique();
  return *this;
}

/// \brief set the angle/axis to identity
template<typename Scalar>
AngleAxisTemplate<Scalar>& AngleAxisTemplate<Scalar>::setIdentity() {
  C_A_B_ = C_A_B_.Identity();
  return *this;
}

/// \brief get a copy of the rotation inverted.
template<typename Scalar>
AngleAxisTemplate<Scalar> AngleAxisTemplate<Scalar>::inverted() const {
  return AngleAxisTemplate(C_A_B_.inverse());
}

/// \brief rotate a vector, v
template<typename Scalar>
typename AngleAxisTemplate<Scalar>::Vector3 AngleAxisTemplate<Scalar>::rotate(
    const AngleAxisTemplate<Scalar>::Vector3& v) const {
  return C_A_B_*v;
}

/// \brief rotate a vector, v
template<typename Scalar>
typename AngleAxisTemplate<Scalar>::Vector4
AngleAxisTemplate<Scalar>::rotate4(
    const AngleAxisTemplate<Scalar>::Vector4& v) const {
  AngleAxisTemplate<Scalar>::Vector4 vprime;
  vprime[3] = v[3];
  vprime.template head<3>() = C_A_B_ * v.template head<3>();
  return vprime;
}

/// \brief rotate a vector, v
template<typename Scalar>
typename AngleAxisTemplate<Scalar>::Vector3
AngleAxisTemplate<Scalar>::inverseRotate(
    const AngleAxisTemplate<Scalar>::Vector3& v) const {
  return C_A_B_.inverse() * v;
}

/// \brief rotate a vector, v
template<typename Scalar>
typename AngleAxisTemplate<Scalar>::Vector4
AngleAxisTemplate<Scalar>::inverseRotate4(
    const typename AngleAxisTemplate<Scalar>::Vector4& v) const {
  Eigen::Vector4d vprime;
  vprime[3] = v[3];
  vprime.template head<3>() = C_A_B_.inverse() * v.template head<3>();
  return vprime;
}

/// \brief cast to the implementation type
template<typename Scalar>
typename AngleAxisTemplate<Scalar>::Implementation&
AngleAxisTemplate<Scalar>::toImplementation() {
  return C_A_B_;
}

/// \brief cast to the implementation type
template<typename Scalar>
const typename AngleAxisTemplate<Scalar>::Implementation&
AngleAxisTemplate<Scalar>::toImplementation() const {
  return C_A_B_;
}

/// \brief enforce the unit length constraint
template<typename Scalar>
AngleAxisTemplate<Scalar>& AngleAxisTemplate<Scalar>::normalize() {
  C_A_B_.axis().normalize();
  return *this;
}

/// \brief compose two rotations
template<typename Scalar>
AngleAxisTemplate<Scalar> AngleAxisTemplate<Scalar>::operator*(
    const AngleAxisTemplate& rhs) const {
  return AngleAxisTemplate(Implementation(C_A_B_ * rhs.C_A_B_));
}

/// \brief get the angle between this and the other rotation
template<typename Scalar>
Scalar AngleAxisTemplate<Scalar>::getDisparityAngle(
    const AngleAxisTemplate& rhs) const {
  return (rhs * this->inverted()).getUnique().angle();
}

template<typename Scalar>
std::ostream& operator<<(std::ostream& out,
                         const AngleAxisTemplate<Scalar>& rhs) {
  out << rhs.vector().transpose();
  return out;
}

/// \brief get the rotation matrix
template<typename Scalar>
typename AngleAxisTemplate<Scalar>::RotationMatrix
AngleAxisTemplate<Scalar>::getRotationMatrix() const {
  return C_A_B_.matrix();
}

} // namespace minimal
} // namespace kindr
#endif  // KINDR_MIN_ROTATION_ANGLE_AXIS_INL_H_
