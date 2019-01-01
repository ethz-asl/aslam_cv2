#ifndef ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_H_
#define ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_H_

/// \addtogroup Matching
/// @{
///
/// @}

#include <map>
#include <memory>
#include <vector>

#include <aslam/common/macros.h>
#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <aslam/common-private/feature-descriptor-ref.h>
#include <Eigen/Core>

#include "aslam/matcher/match.h"
#include "aslam/matcher/matching-problem.h"

namespace aslam {
class VisualFrame;

template <class Scalar>
using Descriptor = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <class Scalar>
struct LandmarkWithDescriptor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LandmarkWithDescriptor() = delete;
  LandmarkWithDescriptor(const Eigen::Vector3d& p_C_landmark, const Descriptor<Scalar>& descriptor,
                         const size_t descriptor_observation_index)
    : p_C_landmark_(p_C_landmark), descriptor_(descriptor),
      descriptor_observation_index_(descriptor_observation_index) {}
  virtual ~LandmarkWithDescriptor() = default;

  const Eigen::Vector3d& get_p_C_landmark() const {
    return p_C_landmark_;
  }

  const Descriptor<Scalar>& getDescriptor() const {
    return descriptor_;
  }

  size_t getDescriptorObservationIndex() const {
    return descriptor_observation_index_;
  }

 private:
  Eigen::Vector3d p_C_landmark_;
  Descriptor<Scalar> descriptor_;
  size_t descriptor_observation_index_;
};

template <class Scalar>
using LandmarkWithDescriptorList = typename Aligned<std::vector, LandmarkWithDescriptor<Scalar>>::type;

/// \class MatchingProblem
/// \brief Defines the specifics of a matching problem.
/// The problem is assumed to have a list of landmarks (3D points + descriptors)
/// and a visual frame with keypoints and descriptors. The landmarks are then
/// matched against the keypoints in the frame based on image space distance
/// and descriptor distance. The landmarks are expected to be expressed in the
/// camera frame of the visual frame.
///
/// Coordinate Frames:
///   C: camera frame of the visual frame.
template <class Scalar>
class MatchingProblemLandmarksToFrame : public MatchingProblem {
public:
  ASLAM_POINTER_TYPEDEFS(MatchingProblemLandmarksToFrame);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingProblemLandmarksToFrame);

  MatchingProblemLandmarksToFrame() = delete;

  /// \brief Constructor for a landmarks-to-frame matching problem.
  ///
  /// @param[in]  frame                                       Visual frame.
  /// @param[in]  landmarks                                   List of landmarks with descriptors.
  /// @param[in]  image_space_distance_threshold_pixels       Max image space distance threshold
  ///                                                         for a keypoint and a projected
  ///                                                         land to become match candidates.
  /// @param[in]  hamming_distance_threshold                  Max hamming distance for a keypoint
  ///                                                         and a projected landmark
  ///                                                         to become candidates.
  MatchingProblemLandmarksToFrame(const VisualFrame& frame,
                                  const LandmarkWithDescriptorList<Scalar>& landmarks,
                                  double image_space_distance_threshold_pixels,
                                  double descriptor_distance_threshold);
  virtual ~MatchingProblemLandmarksToFrame() {};

  virtual size_t numApples() const;
  virtual size_t numBananas() const;

protected:
  inline double computeMatchScore(const double descriptor_distance) const;

  inline double computeDescriptorDistance(int landmark_index, int frame_keypoint_index) const;

  inline int computeHammingDistance(int landmark_index, int frame_keypoint_index) const {
    CHECK_LT(frame_keypoint_index, static_cast<int>(frame_descriptors_.size()))
        << "No descriptor for keypoint with index " << frame_keypoint_index << ".";
    CHECK_LT(landmark_index, static_cast<int>(landmark_descriptors_.size()))
        << "No descriptor for landmark with index " << landmark_index << ".";
    CHECK_LT(frame_keypoint_index, static_cast<int>(is_frame_keypoint_valid_.size()))
        << "No valid flag for keypoint with index " << frame_keypoint_index << ".";
    CHECK_LT(landmark_index, static_cast<int>(is_landmark_valid_.size()))
        << "No valid flag for landmark with index " << landmark_index << ".";
    CHECK(is_frame_keypoint_valid_[frame_keypoint_index])
        << "The given frame keypoint with index " << frame_keypoint_index
        << " is not valid.";
    CHECK(is_landmark_valid_[landmark_index]) << "The given landmark with index "
        << landmark_index << " is not valid.";

    const common::FeatureDescriptorConstRef& apple_descriptor =
        frame_descriptors_[frame_keypoint_index];
    const common::FeatureDescriptorConstRef& banana_descriptor =
        landmark_descriptors_[landmark_index];

    CHECK_NOTNULL(apple_descriptor.data());
    CHECK_NOTNULL(banana_descriptor.data());

    return common::GetNumBitsDifferent(banana_descriptor, apple_descriptor);
  }

  inline double computeL2Norm(int landmark_index, int frame_keypoint_index) const {
    CHECK_LT(frame_keypoint_index, frame_float_descriptors_.size())
        << "No descriptor for keypoint with index " << frame_keypoint_index << ".";
    CHECK_LT(landmark_index, static_cast<int>(landmark_float_descriptors_.size()))
        << "No descriptor for landmark with index " << landmark_index << ".";
    CHECK_LT(frame_keypoint_index, static_cast<int>(is_frame_keypoint_valid_.size()))
        << "No valid flag for keypoint with index " << frame_keypoint_index << ".";
    CHECK_LT(landmark_index, static_cast<int>(is_landmark_valid_.size()))
        << "No valid flag for landmark with index " << landmark_index << ".";
    CHECK(is_frame_keypoint_valid_[frame_keypoint_index])
        << "The given frame keypoint with index " << frame_keypoint_index
        << " is not valid.";
    CHECK(is_landmark_valid_[landmark_index]) << "The given landmark with index "
        << landmark_index << " is not valid.";

    const double l2_norm = (frame_float_descriptors_[frame_keypoint_index] -
        landmark_float_descriptors_[landmark_index]).norm();
    return l2_norm;
  }

  void setupValidVectorsAndDescriptors();

  /// The landmarks to be matched.
  LandmarkWithDescriptorList<Scalar> landmarks_;
  /// The visual frame the landmarks are matched against.
  const VisualFrame& frame_;

  /// Index marking frame keypoints as valid or invalid.
  std::vector<unsigned char> is_frame_keypoint_valid_;
  /// Index marking landmarks as valid or invalid.
  std::vector<unsigned char> is_landmark_valid_;

  /// The frame keypoint descriptors.
  std::vector<common::FeatureDescriptorConstRef> frame_descriptors_;
  // Frame float descriptors, column major.
  Aligned<std::vector, Eigen::Matrix<float, Eigen::Dynamic, 1>>::type frame_float_descriptors_;

  /// The landmark descriptors.
  std::vector<common::FeatureDescriptorConstRef> landmark_descriptors_;
  // Landmark float descriptors, column major.
  Aligned<std::vector, Eigen::Matrix<float, Eigen::Dynamic, 1>>::type landmark_float_descriptors_;

  /// Descriptor size in bits and bytes.
  const size_t descriptor_num_elements_;

  /// The height of the visual frame.
  size_t image_height_;

  /// Pairs with image space distance >= image_space_distance_threshold_pixels_ are
  /// excluded from matches.
  double squared_image_space_distance_threshold_px_sq_;

  /// Pairs with descriptor distance >= hamming_distance_threshold_ are
  /// excluded from matches.
  //int hamming_distance_threshold_;
  double descriptor_distance_threshold_;
};

template <>
inline double MatchingProblemLandmarksToFrame<unsigned char>::computeDescriptorDistance(
    int landmark_index, int frame_keypoint_index) const {
  return static_cast<double>(computeHammingDistance(landmark_index, frame_keypoint_index));
}

template <>
inline double MatchingProblemLandmarksToFrame<float>::computeDescriptorDistance(
    int landmark_index, int frame_keypoint_index) const {
  return static_cast<double>(computeL2Norm(landmark_index, frame_keypoint_index));
}

template <>
inline double MatchingProblemLandmarksToFrame<unsigned char>::computeMatchScore(
    double hamming_distance) const {
  return static_cast<double>(descriptor_num_elements_ * 8u) - hamming_distance /
      static_cast<double>(descriptor_num_elements_ * 8u);
}

template <>
inline double MatchingProblemLandmarksToFrame<float>::computeMatchScore(
    double l2_norm) const {
  return (descriptor_distance_threshold_ - l2_norm) /
      descriptor_distance_threshold_;
}
}  // namespace aslam

#include "./matching-problem-landmarks-to-frame-inl.h"

#endif  // ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_H_
