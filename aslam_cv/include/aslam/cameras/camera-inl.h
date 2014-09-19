namespace aslam {

template<typename DerivedCamera, typename DerivedDistortion>
std::shared_ptr<Camera> Camera::construct(
    const Eigen::VectorXd& intrinsics,
    uint32_t imageWidth,
    uint32_t imageHeight,
    const Eigen::VectorXd& distortionParameters) {
  std::shared_ptr<DerivedDistortion> distortion(new DerivedDistortion(distortionParameters));
  std::shared_ptr<Camera> camera(new DerivedCamera(intrinsics, imageWidth, imageHeight, distortion));
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
