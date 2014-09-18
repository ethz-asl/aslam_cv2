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

}  // namespace aslam
