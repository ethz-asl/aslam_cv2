#include <kindr/minimal/QuatTransformation.h>

namespace kindr {
namespace minimal {

QuatTransformation::QuatTransformation() {

}


QuatTransformation::QuatTransformation(const RotationQuaternion& rotation, const Position& translation) :
    rotation_(rotation),
    translation_(translation) {

}

  
QuatTransformation::~QuatTransformation() {

}


/// \brief get the position component
QuatTransformation::Position& QuatTransformation::getPosition() {
  return translation_;
}

  
/// \brief get the position component
const QuatTransformation::Position& QuatTransformation::getPosition() const {
  return translation_;
}


/// \brief get the rotation component
QuatTransformation::Rotation& QuatTransformation::getRotation() {
  return rotation_;
}

  
/// \brief get the rotation component
const QuatTransformation::Rotation& QuatTransformation::getRotation() const {
  return rotation_;
}

  
/// \brief get the transformation matrix
QuatTransformation::TransformationMatrix QuatTransformation::getTransformationMatrix() const {
  TransformationMatrix T;
  T.setIdentity();
  T.topLeftCorner<3,3>() = rotation_.getRotationMatrix();
  T.topRightCorner<3,1>() = translation_;
  return T;
}


/// \brief compose two transformations
QuatTransformation QuatTransformation::operator*(const QuatTransformation& rhs) const {
  return QuatTransformation( rotation_ * rhs.rotation_, translation_ + rotation_.rotate(rhs.translation_));
}


/// \brief transform a point
Eigen::Vector3d QuatTransformation::transform(const Eigen::Vector3d& rhs) const {
  return rotation_.rotate(rhs) + translation_;
}


/// \brief transform a point
Eigen::Vector4d QuatTransformation::transform4(const Eigen::Vector4d& rhs) const {
  Eigen::Vector4d rval;
  rval[3] = rhs[3];
  rval.head<3>() = rotation_.rotate(rhs.head<3>()) + rhs[3]*translation_;
  return rval;
}


/// \brief transform a vector (apply only the rotational component)
Eigen::Vector3d QuatTransformation::transformVector(const Eigen::Vector3d& rhs) const {
  return rotation_.rotate(rhs);
}


/// \brief transform a point by the inverse
Eigen::Vector3d QuatTransformation::inverseTransform(const Eigen::Vector3d& rhs) const {
  return rotation_.inverseRotate(rhs - translation_);
}

/// \brief transform a point by the inverse
Eigen::Vector4d QuatTransformation::inverseTransform4(const Eigen::Vector4d& rhs) const {
  Eigen::Vector4d rval;
  rval.head<3>() = rotation_.inverseRotate(rhs.head<3>() - translation_*rhs[3]);
  rval[3] = rhs[3];
  return rval;
}

  /// \brief transform a vector by the inverse (apply only the rotational component)
Eigen::Vector3d QuatTransformation::inverseTransformVector(const Eigen::Vector3d& rhs) const {
  return rotation_.inverseRotate(rhs);
}


/// \brief return a copy of the transformation inverted
QuatTransformation QuatTransformation::inverted() const {
  return QuatTransformation(rotation_.inverted(), -rotation_.inverseRotate(translation_));
}

/// \brief invert the transformation
QuatTransformation& QuatTransformation::invert(){
  rotation_.invert();
  translation_ = -rotation_.rotate(translation_);
  return *this;
}



std::ostream & operator<<(std::ostream & out, const QuatTransformation& pose) {
  out << pose.getTransformationMatrix();
  return out;
}

} // namespace minimal
} // namespace kindr
