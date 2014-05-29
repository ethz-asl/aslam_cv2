#ifndef ASLAM_FRAMES_VISUAL_FRAME_H_
#define ASLAM_FRAMES_VISUAL_FRAME_H_

#include <memory>
#include <aslam/common/macros.h>
#include <Eigen/Dense>

namespace aslam {
namespace cameras {
class Camera;
}

class VisualFrame : public FrameBase {
 public:
  typedef Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> DescriptorsT;
  ASLAM_POINTER_TYPEDEFS(VisualFrame);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual bool operator==(const VisualFrame& other) const;
  /// \brief The keypoints stored in a frame.
  const Eigen::Matrix2Xd& getKeypoints() const;
  /// \brief The descriptors stored in a frame.
  const DescriptorsT& getDescriptors() const;

  /// \brief A pointer to the keypoints, can be used to swap in new data.
  Eigen::Matrix2Xd* getKeypointsMutable();
  /// \brief A pointer to the descriptors, can be used to swap in new data.
  DescriptorsT* getDescriptorsMutable();

  /// \brief Return pointer location of the descriptor pointed to by index.
  const char* getDescriptor(size_t index) const;
  /// \brief Return block expression of the keypoint pointed to by index.
  const Eigen::Block<Eigen::Matrix2Xd, 2, 1> getKeypoint(size_t index) const;

  /// \brief Replace (copy) the internal keypoints by the passed ones.
  const void setKeypoints(const Eigen::Matrix2Xd& keypoints);
  /// \brief Replace (copy) the internal descriptors by the passed ones.
  const void setDescriptors(const DescriptorsT& descriptors);

  /// \brief The camera geometry.
  const std::shared_ptr<const cameras::Camera> getCameraGeometry() const;
  /// \brief Set the camera geometry.
  void setCameraGeometry(const std::shared_ptr<cameras::Camera>& camera);

 private:
  unordered_map<std::string, std::shared_ptr<aslam::ChannelBase> > channels_;
  std::shared_ptr<cameras::Camera> camera_geometry_;
};
}  // namespace aslam
#endif  // ASLAM_FRAMES_VISUAL_FRAME_H_
