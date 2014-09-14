#include <kindr/minimal/QuatTransformation.h>

namespace kindr {
namespace minimal {

QuatTransformation::QuatTransformation() {
  setIdentity();
}


QuatTransformation::QuatTransformation(const RotationQuaternion& q_A_B, const Position& A_t_A_B) :
    q_A_B_(q_A_B),
    A_t_A_B_(A_t_A_B) {

}

QuatTransformation::QuatTransformation(const Rotation::Implementation& q_A_B,
                                       const Position& A_t_A_B) :
        q_A_B_(q_A_B),
        A_t_A_B_(A_t_A_B) {

}

QuatTransformation::QuatTransformation(const Position& A_t_A_B,
                                       const RotationQuaternion& q_A_B) :
    q_A_B_(q_A_B),
    A_t_A_B_(A_t_A_B) {

}

QuatTransformation::QuatTransformation(const Position& A_t_A_B,
                                       const Rotation::Implementation& q_A_B) :
        q_A_B_(q_A_B),
        A_t_A_B_(A_t_A_B) {

}

QuatTransformation::~QuatTransformation() {

}

void QuatTransformation::setIdentity() {
  q_A_B_.setIdentity();
  A_t_A_B_.setZero();
}

/// \brief get the position component
QuatTransformation::Position& QuatTransformation::getPosition() {
  return A_t_A_B_;
}

  
/// \brief get the position component
const QuatTransformation::Position& QuatTransformation::getPosition() const {
  return A_t_A_B_;
}


/// \brief get the rotation component
QuatTransformation::Rotation& QuatTransformation::getRotation() {
  return q_A_B_;
}

  
/// \brief get the rotation component
const QuatTransformation::Rotation& QuatTransformation::getRotation() const {
  return q_A_B_;
}

  
/// \brief get the transformation matrix
QuatTransformation::TransformationMatrix QuatTransformation::getTransformationMatrix() const {
  TransformationMatrix T;
  T.setIdentity();
  T.topLeftCorner<3,3>() = q_A_B_.getRotationMatrix();
  T.topRightCorner<3,1>() = A_t_A_B_;
  return T;
}


/// \brief compose two transformations
QuatTransformation QuatTransformation::operator*(const QuatTransformation& rhs) const {
  return QuatTransformation( q_A_B_ * rhs.q_A_B_, A_t_A_B_ + q_A_B_.rotate(rhs.A_t_A_B_));
}


/// \brief transform a point
Eigen::Vector3d QuatTransformation::transform(const Eigen::Vector3d& rhs) const {
  return q_A_B_.rotate(rhs) + A_t_A_B_;
}


/// \brief transform a point
Eigen::Vector4d QuatTransformation::transform4(const Eigen::Vector4d& rhs) const {
  Eigen::Vector4d rval;
  rval[3] = rhs[3];
  rval.head<3>() = q_A_B_.rotate(rhs.head<3>()) + rhs[3]*A_t_A_B_;
  return rval;
}


/// \brief transform a vector (apply only the rotational component)
Eigen::Vector3d QuatTransformation::transformVector(const Eigen::Vector3d& rhs) const {
  return q_A_B_.rotate(rhs);
}


/// \brief transform a point by the inverse
Eigen::Vector3d QuatTransformation::inverseTransform(const Eigen::Vector3d& rhs) const {
  return q_A_B_.inverseRotate(rhs - A_t_A_B_);
}

/// \brief transform a point by the inverse
Eigen::Vector4d QuatTransformation::inverseTransform4(const Eigen::Vector4d& rhs) const {
  Eigen::Vector4d rval;
  rval.head<3>() = q_A_B_.inverseRotate(rhs.head<3>() - A_t_A_B_*rhs[3]);
  rval[3] = rhs[3];
  return rval;
}

  /// \brief transform a vector by the inverse (apply only the rotational component)
Eigen::Vector3d QuatTransformation::inverseTransformVector(const Eigen::Vector3d& rhs) const {
  return q_A_B_.inverseRotate(rhs);
}


/// \brief return a copy of the transformation inverted
QuatTransformation QuatTransformation::inverted() const {
  return QuatTransformation(q_A_B_.inverted(), -q_A_B_.inverseRotate(A_t_A_B_));
}


std::ostream & operator<<(std::ostream & out, const QuatTransformation& pose) {
  out << pose.getTransformationMatrix();
  return out;
}

/// \brief check for binary equality
bool QuatTransformation::operator==(const QuatTransformation& rhs) const {
  return q_A_B_ == rhs.q_A_B_ && A_t_A_B_ == rhs.A_t_A_B_;
}


} // namespace minimal
} // namespace kindr
