#ifndef KINDR_MINIMAL_QUAT_TRANSFORMATION_H_INL_
#define KINDR_MINIMAL_QUAT_TRANSFORMATION_H_INL_
#include <kindr/minimal/quat-transformation.h>

namespace kindr {
namespace minimal {

template<typename Scalar>
QuatTransformationTemplate<Scalar>::QuatTransformationTemplate() {
  setIdentity();
}

template<typename Scalar>
QuatTransformationTemplate<Scalar>::QuatTransformationTemplate(
    const RotationQuaternionTemplate<Scalar>& q_A_B, const Position& A_t_A_B) :
    q_A_B_(q_A_B),
    A_t_A_B_(A_t_A_B) {

}

template<typename Scalar>
QuatTransformationTemplate<Scalar>::QuatTransformationTemplate(
    const typename Rotation::Implementation& q_A_B,
    const Position& A_t_A_B) :
        q_A_B_(q_A_B),
        A_t_A_B_(A_t_A_B) {

}

template<typename Scalar>
QuatTransformationTemplate<Scalar>::QuatTransformationTemplate(
    const Position& A_t_A_B, const RotationQuaternionTemplate<Scalar>& q_A_B) :
    q_A_B_(q_A_B),
    A_t_A_B_(A_t_A_B) {

}

template<typename Scalar>
QuatTransformationTemplate<Scalar>::QuatTransformationTemplate(
    const Position& A_t_A_B, const typename Rotation::Implementation& q_A_B) :
        q_A_B_(q_A_B),
        A_t_A_B_(A_t_A_B) {

}

template<typename Scalar>
QuatTransformationTemplate<Scalar>::QuatTransformationTemplate(
    const TransformationMatrix& T) :
    q_A_B_(T.template topLeftCorner<3,3>().eval()),
    A_t_A_B_(T.template topRightCorner<3,1>().eval()) {
}

/// \brief a constructor based on the exponential map
/// translational part in the first 3 dimensions, 
/// rotational part in the last 3 dimensions
template<typename Scalar>
QuatTransformationTemplate<Scalar>::QuatTransformationTemplate(
     const QuatTransformationTemplate<Scalar>::Vector6& x_t_r) : 
  q_A_B_(x_t_r.template tail<3>().eval()),
  A_t_A_B_(x_t_r.template head<3>().eval()) {
}
                                                              
template<typename Scalar>
QuatTransformationTemplate<Scalar>::~QuatTransformationTemplate() {

}

template<typename Scalar>
void QuatTransformationTemplate<Scalar>::setIdentity() {
  q_A_B_.setIdentity();
  A_t_A_B_.setZero();
}

/// \brief get the position component
template<typename Scalar>
typename QuatTransformationTemplate<Scalar>::Position&
QuatTransformationTemplate<Scalar>::getPosition() {
  return A_t_A_B_;
}

  
/// \brief get the position component
template<typename Scalar>
const typename QuatTransformationTemplate<Scalar>::Position&
QuatTransformationTemplate<Scalar>::getPosition() const {
  return A_t_A_B_;
}


/// \brief get the rotation component
template<typename Scalar>
typename QuatTransformationTemplate<Scalar>::Rotation&
QuatTransformationTemplate<Scalar>::getRotation() {
  return q_A_B_;
}

  
/// \brief get the rotation component
template<typename Scalar>
const typename QuatTransformationTemplate<Scalar>::Rotation&
QuatTransformationTemplate<Scalar>::getRotation() const {
  return q_A_B_;
}

  
/// \brief get the transformation matrix
template<typename Scalar>
typename QuatTransformationTemplate<Scalar>::TransformationMatrix
QuatTransformationTemplate<Scalar>::getTransformationMatrix() const {
  TransformationMatrix transformation_matrix;
  transformation_matrix.setIdentity();
  transformation_matrix.template topLeftCorner<3,3>() =
      q_A_B_.getRotationMatrix();
  transformation_matrix.template topRightCorner<3,1>() = A_t_A_B_;
  return transformation_matrix;
}


/// \brief compose two transformations
template<typename Scalar>
QuatTransformationTemplate<Scalar>
QuatTransformationTemplate<Scalar>::operator*(
    const QuatTransformationTemplate<Scalar>& rhs) const {
  return QuatTransformationTemplate<Scalar>(q_A_B_ * rhs.q_A_B_, A_t_A_B_ +
                                            q_A_B_.rotate(rhs.A_t_A_B_));
}


/// \brief transform a point
template<typename Scalar>
typename QuatTransformationTemplate<Scalar>::Vector3
QuatTransformationTemplate<Scalar>::transform(
    const typename QuatTransformationTemplate<Scalar>::Vector3& rhs) const {
  return q_A_B_.rotate(rhs) + A_t_A_B_;
}


/// \brief transform a point
template<typename Scalar>
typename QuatTransformationTemplate<Scalar>::Vector4
QuatTransformationTemplate<Scalar>::transform4(
    const typename QuatTransformationTemplate<Scalar>::Vector4& rhs) const {
  QuatTransformationTemplate<Scalar>::Vector4 rval;
  rval[3] = rhs[3];
  rval.template head<3>() =
      q_A_B_.rotate(rhs.template head<3>()) + rhs[3]*A_t_A_B_;
  return rval;
}


/// \brief transform a vector (apply only the rotational component)
template<typename Scalar>
typename QuatTransformationTemplate<Scalar>::Vector3
QuatTransformationTemplate<Scalar>::transformVector(
    const QuatTransformationTemplate<Scalar>::Vector3& rhs) const {
  return q_A_B_.rotate(rhs);
}


/// \brief transform a point by the inverse
template<typename Scalar>
typename QuatTransformationTemplate<Scalar>::Vector3
QuatTransformationTemplate<Scalar>::inverseTransform(
    const Vector3& rhs) const {
  return q_A_B_.inverseRotate(rhs - A_t_A_B_);
}

/// \brief transform a point by the inverse
template<typename Scalar>
typename QuatTransformationTemplate<Scalar>::Vector4
QuatTransformationTemplate<Scalar>::inverseTransform4(
    const typename QuatTransformationTemplate<Scalar>::Vector4& rhs) const {
  typename QuatTransformationTemplate<Scalar>::Vector4 rval;
  rval.template head<3>() = q_A_B_.inverseRotate(rhs.template head<3>() -
                                                 A_t_A_B_*rhs[3]);
  rval[3] = rhs[3];
  return rval;
}

/// \brief transform a vector by the inverse (apply only the rotational
///        component)
template<typename Scalar>
typename QuatTransformationTemplate<Scalar>::Vector3
QuatTransformationTemplate<Scalar>::inverseTransformVector(
    const typename QuatTransformationTemplate<Scalar>::Vector3& rhs) const {
  return q_A_B_.inverseRotate(rhs);
}


/// \brief return a copy of the transformation inverted
template<typename Scalar>
QuatTransformationTemplate<Scalar>
QuatTransformationTemplate<Scalar>::inverted() const {
  return QuatTransformation(q_A_B_.inverted(), -q_A_B_.inverseRotate(A_t_A_B_));
}

/// \brief get the logarithmic map of the transformation
/// note: this is the log map of SO(3)xR(3) and not SE(3)
/// \return vector form of log map with first 3 components the translational part and the last three the rotational part.
template<typename Scalar>
typename QuatTransformationTemplate<Scalar>::Vector6 
QuatTransformationTemplate<Scalar>::log() const {
  AngleAxisTemplate<Scalar> angleaxis(q_A_B_);
  return (Vector6() << A_t_A_B_, (angleaxis.axis()*angleaxis.angle())).finished();
}


template<typename Scalar>
std::ostream & operator<<(std::ostream & out,
                          const QuatTransformationTemplate<Scalar>& pose) {
  out << pose.getTransformationMatrix();
  return out;
}

/// \brief check for binary equality
template<typename Scalar>
bool QuatTransformationTemplate<Scalar>::operator==(
    const QuatTransformationTemplate<Scalar>& rhs) const {
  return q_A_B_ == rhs.q_A_B_ && A_t_A_B_ == rhs.A_t_A_B_;
}


} // namespace minimal
} // namespace kindr
#endif  // KINDR_MINIMAL_QUAT_TRANSFORMATION_H_INL_
