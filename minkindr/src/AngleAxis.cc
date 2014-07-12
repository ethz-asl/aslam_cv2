#include <kindr/minimal/AngleAxis.h>
#include <kindr/minimal/RotationQuaternion.h>
#include <glog/logging.h>

namespace kindr {
namespace minimal {

/// \brief initialize to identity
AngleAxis::AngleAxis() : 
    C_A_B_(Implementation::Identity()) {
}

/// \brief initialize from real and imaginary components (real first)
AngleAxis::AngleAxis(Scalar w, Scalar x, Scalar y, Scalar z) :
    C_A_B_(w,Vector3(x,y,z)) {
  CHECK_NEAR(Vector3(x,y,z).squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
}
 
/// \brief initialize from real and imaginary components
AngleAxis::AngleAxis(Scalar angle, const Eigen::Vector3d& axis) :
    C_A_B_(angle, axis){
    CHECK_NEAR(axis.squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
}

/// \brief initialize from an Eigen angleAxis
AngleAxis::AngleAxis(const Implementation& angleAxis) :
    C_A_B_(angleAxis) {
}

/// \brief initialize from a rotation matrix
AngleAxis::AngleAxis(const RotationMatrix& matrix) :
    C_A_B_(matrix) {
  // \todo furgalep Check that the matrix was good...
}


AngleAxis::AngleAxis(const RotationQuaternion& quat) :
    C_A_B_(quat.toImplementation()) {
}
  
AngleAxis::~AngleAxis() { }

/// \brief assignment operator
AngleAxis& AngleAxis::operator=(const AngleAxis& rhs) {
  if(this != &rhs) {
    C_A_B_ = rhs.C_A_B_;
  }
  return *this;
}

  /// \brief Returns the rotation angle.
AngleAxis::Scalar AngleAxis::angle() const{
  return C_A_B_.angle();
}

/// \brief Sets the rotation angle.
void AngleAxis::setAngle(Scalar angle){
  C_A_B_.angle() = angle;
}

/// \brief Returns the rotation axis.
const AngleAxis::Vector3& AngleAxis::axis() const{
  return C_A_B_.axis();
}

/// \brief Sets the rotation axis.
void AngleAxis::setAxis(const Vector3& axis){
  CHECK_NEAR(axis.squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
  C_A_B_.axis() = axis;
}

/// \brief Sets the rotation axis.
void AngleAxis::setAxis(Scalar v1, Scalar v2, Scalar v3){
  C_A_B_.axis() = Vector3(v1,v2,v3);
  CHECK_NEAR(C_A_B_.axis().squaredNorm(), static_cast<Scalar>(1.0), static_cast<Scalar>(1e-4));
}

/// \brief get the components of the angle/axis as a vector (real first)
AngleAxis::Vector4 AngleAxis::vector() const{
    Vector4 vector;
    vector(0) = angle();
    vector.tail<3>() = C_A_B_.axis();
    return vector;
}

/// \brief get a copy of the representation that is unique
AngleAxis AngleAxis::getUnique() const {
  AngleAxis aa(fmod(angle()+M_PI, 2*M_PI)-M_PI, C_A_B_.axis()); // first wraps angle into [-pi,pi)
    if(aa.angle() > 0)  {
      return aa;
    } else if(aa.angle() < 0) {
      if(aa.angle() != -M_PI) {
        return AngleAxis(-aa.angle(), -aa.axis());
      } else { // angle == -pi, so axis must be viewed further, because -pi,axis does the same as -pi,-axis

        if(aa.axis()[0] < 0) {
          return AngleAxis(-aa.angle(), -aa.axis());
        } else if(aa.axis()[0] > 0) {
          return AngleAxis(-aa.angle(), aa.axis());
        } else { // v1 == 0
          if(aa.axis()[1] < 0) {
            return AngleAxis(-aa.angle(), -aa.axis());
          } else if(aa.axis()[1] > 0) {
            return AngleAxis(-aa.angle(), aa.axis());
          } else { // v2 == 0
            if(aa.axis()[2] < 0) { // v3 must be -1 or 1
              return AngleAxis(-aa.angle(), -aa.axis());
            } else  {
              return AngleAxis(-aa.angle(), aa.axis());
            }
          }
        }
      }
    } else { // angle == 0
      return AngleAxis();
    }
}

/// \brief set the angle/axis to its unique representation
AngleAxis& AngleAxis::setUnique() {
  *this = getUnique();
  return *this;
}

/// \brief set the angle/axis to identity
AngleAxis& AngleAxis::setIdentity() {
  C_A_B_ = C_A_B_.Identity();
  return *this;
}

/// \brief invert the rotation
AngleAxis& AngleAxis::invert() {
  C_A_B_ = C_A_B_.inverse();
  return *this;
}

/// \brief get a copy of the rotation inverted.
AngleAxis AngleAxis::inverted() const {
  return AngleAxis(C_A_B_.inverse());
}

/// \brief rotate a vector, v
Eigen::Vector3d AngleAxis::rotate(const Eigen::Vector3d& v) const {
  return C_A_B_*v;
}

/// \brief rotate a vector, v
Eigen::Vector4d AngleAxis::rotate4(const Eigen::Vector4d& v) const {
  Eigen::Vector4d vprime;
  vprime[3] = v[3];
  vprime.head<3>() = C_A_B_*v.head<3>();
  return vprime;
}

/// \brief rotate a vector, v
Eigen::Vector3d AngleAxis::inverseRotate(const Eigen::Vector3d& v) const {
  return C_A_B_.inverse()*v;
}

/// \brief rotate a vector, v
Eigen::Vector4d AngleAxis::inverseRotate4(const Eigen::Vector4d& v) const {
  Eigen::Vector4d vprime;
  vprime[3] = v[3];
  vprime.head<3>() = C_A_B_.inverse()*v.head<3>();
  return vprime;
}

/// \brief cast to the implementation type
AngleAxis::Implementation& AngleAxis::toImplementation() {
  return C_A_B_;
}

/// \brief cast to the implementation type
const AngleAxis::Implementation& AngleAxis::toImplementation() const {
  return C_A_B_;
}

/// \brief enforce the unit length constraint
AngleAxis& AngleAxis::fix() {
  C_A_B_.axis().normalize();
  return *this;
}

/// \brief compose two rotations
AngleAxis AngleAxis::operator*(const AngleAxis& rhs) const {
  return AngleAxis(Implementation(C_A_B_ * rhs.C_A_B_));
}

/// \brief get the angle between this and the other rotation
AngleAxis::Scalar AngleAxis::getDisparityAngle(const AngleAxis& rhs) const {
  return (rhs * this->inverted()).getUnique().angle();
}

std::ostream& operator<<(std::ostream& out, const AngleAxis& rhs) {
  out << rhs.vector().transpose();
  return out;
}

/// \brief get the rotation matrix
AngleAxis::RotationMatrix AngleAxis::getRotationMatrix() const {
  return C_A_B_.matrix();
}

} // namespace minimal
} // namespace kindr
