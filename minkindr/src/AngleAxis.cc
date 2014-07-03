#include <kindr/minimal/AngleAxis.h>
#include <kindr/minimal/RotationQuaternion.h>
#include <glog/logging.h>

namespace kindr {
namespace minimal {

/// \brief initialize to identity
AngleAxis::AngleAxis() : 
    angleAxis_(Implementation::Identity()) {
}


/// \brief initialize from real and imaginary components (real first)
AngleAxis::AngleAxis(Scalar w, Scalar x, Scalar y, Scalar z) :
    angleAxis_(w,Vector3(x,y,z)) {
    CHECK_NEAR(Vector3(x,y,z).squaredNorm(), static_cast<Scalar>(1.0), 1e-4);
}

  
/// \brief initialize from real and imaginary components
AngleAxis::AngleAxis(Scalar angle, const Eigen::Vector3d& axis) :
    angleAxis_(angle, axis){
    CHECK_NEAR(axis.squaredNorm(), static_cast<Scalar>(1.0), 1e-4);
}


/// \brief initialize from an Eigen quaternion
AngleAxis::AngleAxis(const Implementation& angleAxis) :
    angleAxis_(angleAxis) {
}

/// \brief initialize from an Eigen quaternion
AngleAxis::AngleAxis(const RotationMatrix& matrix) :
    angleAxis_(matrix) {
  // \todo furgalep Check that the matrix was good...
}

AngleAxis::AngleAxis(const RotationQuaternion& quat) :
    angleAxis_(quat.toImplementation()) {
  
}
  
AngleAxis::~AngleAxis() {

}

/// \brief assignment operator
AngleAxis& AngleAxis::operator=(const AngleAxis& rhs) {
  if(this != &rhs) {
    angleAxis_ = rhs.angleAxis_;
  }
  return *this;
}


  /// \brief Returns the rotation angle.
AngleAxis::Scalar AngleAxis::angle() const{
  return angleAxis_.angle();
}

/// \brief Sets the rotation angle.
void AngleAxis::setAngle(Scalar angle){
  angleAxis_.angle() = angle;
}

/// \brief Returns the rotation axis.
const AngleAxis::Vector3& AngleAxis::axis() const{
  return angleAxis_.axis();
}

/// \brief Sets the rotation axis.
void AngleAxis::setAxis(const Vector3& axis){
  CHECK_NEAR(axis.squaredNorm(), static_cast<Scalar>(1.0), 1e-4);
  angleAxis_.axis() = axis;
}

/// \brief Sets the rotation axis.
void AngleAxis::setAxis(Scalar v1, Scalar v2, Scalar v3){
  angleAxis_.axis() = Vector3(v1,v2,v3);
  CHECK_NEAR(angleAxis_.axis().squaredNorm(), static_cast<Scalar>(1.0), 1e-4);
}

/// \brief get the components of the quaternion as a vector (real first)
AngleAxis::Vector4 AngleAxis::vector() const{
    Vector4 vector;
    vector(0) = angle();
    vector.tail<3>() = angleAxis_.axis();
    return vector;
}



/// \brief get a copy of the representation that is unique
AngleAxis AngleAxis::getUnique() const {
  AngleAxis aa(fmod(angle()+M_PI,2*M_PI)-M_PI, angleAxis_.axis()); // first wraps angle into [-pi,pi)
    if(aa.angle() > 0)  {
      return aa;
    } else if(aa.angle() < 0) {
      if(aa.angle() != -M_PI) {
        return AngleAxis(-aa.angle(),-aa.axis());
      } else { // angle == -pi, so axis must be viewed further, because -pi,axis does the same as -pi,-axis

        if(aa.axis()[0] < 0) {
          return AngleAxis(-aa.angle(),-aa.axis());
        } else if(aa.axis()[0] > 0) {
          return AngleAxis(-aa.angle(),aa.axis());
        } else { // v1 == 0
          if(aa.axis()[1] < 0) {
            return AngleAxis(-aa.angle(),-aa.axis());
          } else if(aa.axis()[1] > 0) {
            return AngleAxis(-aa.angle(),aa.axis());
          } else { // v2 == 0
            if(aa.axis()[2] < 0) { // v3 must be -1 or 1
              return AngleAxis(-aa.angle(),-aa.axis());
            } else  {
              return AngleAxis(-aa.angle(),aa.axis());
            }
          }
        }
      }
    } else { // angle == 0
      return AngleAxis();
    }
}


/// \brief set the quaternion to its unique representation
AngleAxis& AngleAxis::setUnique() {
  *this = getUnique();
  return *this;
}


/// \brief set the quaternion to identity
AngleAxis& AngleAxis::setIdentity() {
  angleAxis_ = angleAxis_.Identity();
  return *this;
}


/// \brief invert the quaternion
AngleAxis& AngleAxis::invert() {
  angleAxis_ = angleAxis_.inverse();
  return *this;
}


/// \brief get a copy of the quaternion inverted.
AngleAxis AngleAxis::inverted() const {
  return AngleAxis(angleAxis_.inverse());
}


/// \brief rotate a vector, v
Eigen::Vector3d AngleAxis::rotate(const Eigen::Vector3d& v) const {
  return angleAxis_*v;
}


/// \brief rotate a vector, v
Eigen::Vector4d AngleAxis::rotate4(const Eigen::Vector4d& v) const {
  Eigen::Vector4d vprime;
  vprime[3] = v[3];
  vprime.head<3>() = angleAxis_*v.head<3>();
  return vprime;
}


/// \brief rotate a vector, v
Eigen::Vector3d AngleAxis::inverseRotate(const Eigen::Vector3d& v) const {
  return angleAxis_.inverse()*v;
}


/// \brief rotate a vector, v
Eigen::Vector4d AngleAxis::inverseRotate4(const Eigen::Vector4d& v) const {
  Eigen::Vector4d vprime;
  vprime[3] = v[3];
  vprime.head<3>() = angleAxis_.inverse()*v.head<3>();
  return vprime;
}


/// \brief cast to the implementation type
AngleAxis::Implementation& AngleAxis::toImplementation() {
  return angleAxis_;
}


/// \brief cast to the implementation type
const AngleAxis::Implementation& AngleAxis::toImplementation() const {
  return angleAxis_;
}


/// \brief enforce the unit length constraint
AngleAxis& AngleAxis::fix() {
  angleAxis_.axis().normalize();
  return *this;
}


/// \brief compose two quaternions
AngleAxis AngleAxis::operator*(const AngleAxis& rhs) const {
  return AngleAxis(Implementation(angleAxis_ * rhs.angleAxis_));
}


/// \brief get the angle between this and the other quaternion
AngleAxis::Scalar AngleAxis::getDisparityAngle(const AngleAxis& rhs) const {
  return (rhs * this->inverted()).getUnique().angle();
}


std::ostream& operator<<(std::ostream& out, const AngleAxis& rhs) {
  out << rhs.vector().transpose();
  return out;
}


/// \brief get the rotation matrix
AngleAxis::RotationMatrix AngleAxis::getRotationMatrix() const {
  return angleAxis_.matrix();
}


} // namespace minimal
} // namespace kindr
