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

}  // namespace aslam
