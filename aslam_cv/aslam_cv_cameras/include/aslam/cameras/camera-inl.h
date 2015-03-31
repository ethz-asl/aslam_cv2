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

inline bool Camera::isKeypointVisible(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const {
  return keypoint[0] >= 0.0
      && keypoint[1] >= 0.0
      && keypoint[0] < imageWidth()
      && keypoint[1] < imageHeight();
}

inline std::ostream& operator<<(std::ostream& out, const Camera::Type& value) {
  static std::map<Camera::Type, std::string> names;
  if (names.size() == 0) {
    #define INSERT_ELEMENT(type, val) names[type::val] = #val
    INSERT_ELEMENT(Camera::Type, kPinhole);
    INSERT_ELEMENT(Camera::Type, kUnifiedProjection);
    #undef INSERT_ELEMENT
  }
  return out << names[value];
}

}  // namespace aslam
