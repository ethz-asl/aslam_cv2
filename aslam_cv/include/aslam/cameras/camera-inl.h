namespace aslam {

template<typename DerivedCamera, typename DerivedDistortion>
typename DerivedCamera::Ptr Camera::construct(
    const Eigen::VectorXd& intrinsics,
    uint32_t imageWidth,
    uint32_t imageHeight,
    const Eigen::VectorXd& distortionParameters) {
  aslam::Distortion::UniquePtr distortion(new DerivedDistortion(distortionParameters));
  typename DerivedCamera::Ptr camera(new DerivedCamera(intrinsics, imageWidth, imageHeight, distortion));
  return camera;
}

template<typename Scalar>
bool Camera::isKeypointVisible(const Eigen::Matrix<Scalar, 2, 1>& keypoint) const {
  return keypoint[0] >= static_cast<Scalar>(0.0)
      && keypoint[1] >= static_cast<Scalar>(0.0)
      && keypoint[0] < static_cast<Scalar>(imageWidth())
      && keypoint[1] < static_cast<Scalar>(imageHeight());
}

}  // namespace aslam
