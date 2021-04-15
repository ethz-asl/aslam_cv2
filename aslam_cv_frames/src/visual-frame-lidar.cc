#include <aslam/common/channel-definitions-lidar.h>
#include <aslam/common/stl-helpers.h>
#include <aslam/common/time.h>
#include <memory>

#include "aslam/frames/visual-frame.h"

namespace aslam {

bool VisualFrame::hasLidarTrackIds() const {
  return aslam::channels::has_LIDAR_TRACK_IDS_Channel(channels_);
}
bool VisualFrame::hasLidarKeypoint3DMeasurements() const {
  return aslam::channels::has_LIDAR_3D_MEASUREMENTS_Channel(channels_);
}

bool VisualFrame::hasLidarKeypoint2DMeasurements() const {
  return aslam::channels::has_LIDAR_2D_MEASUREMENTS_Channel(channels_);
}

bool VisualFrame::hasLidarDescriptors() const {
  return aslam::channels::has_LIDAR_DESCRIPTORS_Channel(channels_);
}

bool VisualFrame::hasLidarKeypoint2DMeasurementUncertainties() const {
  return aslam::channels::has_LIDAR_KEYPOINT_2D_MEASUREMENT_UNCERTAINTIES_Channel(
      channels_);
}

const Eigen::VectorXi& VisualFrame::getLidarTrackIds() const {
  return aslam::channels::get_LIDAR_TRACK_IDS_Data(channels_);
}

const Eigen::Matrix3Xd& VisualFrame::getLidarKeypoint3DMeasurements() const {
  return aslam::channels::get_LIDAR_3D_MEASUREMENTS_Data(channels_);
}

const Eigen::Matrix2Xd& VisualFrame::getLidarKeypoint2DMeasurements() const {
  return aslam::channels::get_LIDAR_2D_MEASUREMENTS_Data(channels_);
}

const VisualFrame::DescriptorsT& VisualFrame::getLidarDescriptors() const {
  return aslam::channels::get_LIDAR_DESCRIPTORS_Data(channels_);
}

const Eigen::VectorXd& VisualFrame::getLidarKeypoint2DMeasurementUncertainties()
    const {
  return aslam::channels::get_LIDAR_KEYPOINT_2D_MEASUREMENT_UNCERTAINTIES_Data(
      channels_);
}

Eigen::VectorXi* VisualFrame::getLidarTrackIdsMutable() {
  Eigen::VectorXi& lidar_track_ids =
      aslam::channels::get_LIDAR_TRACK_IDS_Data(channels_);
  return &lidar_track_ids;
}

Eigen::Matrix3Xd* VisualFrame::getLidarKeypoint3DMeasurementsMutable() {
  Eigen::Matrix3Xd& vector =
      aslam::channels::get_LIDAR_3D_MEASUREMENTS_Data(channels_);
  return &vector;
}

Eigen::Matrix2Xd* VisualFrame::getLidarKeypoint2DMeasurementsMutable() {
  Eigen::Matrix2Xd& vector =
      aslam::channels::get_LIDAR_2D_MEASUREMENTS_Data(channels_);
  return &vector;
}

VisualFrame::DescriptorsT* VisualFrame::getLidarDescriptorsMutable() {
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_LIDAR_DESCRIPTORS_Data(channels_);
  return &descriptors;
}

Eigen::VectorXd*
VisualFrame::getLidarKeypoint2DMeasurementUncertaintiesMutable() {
  Eigen::VectorXd& uncertainties =
      aslam::channels::get_LIDAR_KEYPOINT_2D_MEASUREMENT_UNCERTAINTIES_Data(
          channels_);
  return &uncertainties;
}

const Eigen::Block<Eigen::Matrix3Xd, 3, 1>
VisualFrame::getLidarKeypoint3DMeasurement(const std::size_t index) const {
  Eigen::Matrix3Xd& keypoints =
      aslam::channels::get_LIDAR_3D_MEASUREMENTS_Data(channels_);
  CHECK_LT(static_cast<int>(index), keypoints.cols());
  return keypoints.block<3, 1>(0, index);
}

const Eigen::Block<Eigen::Matrix2Xd, 2, 1>
VisualFrame::getLidarKeypoint2DMeasurement(const std::size_t index) const {
  Eigen::Matrix2Xd& keypoints =
      aslam::channels::get_LIDAR_2D_MEASUREMENTS_Data(channels_);
  CHECK_LT(static_cast<int>(index), keypoints.cols());
  return keypoints.block<2, 1>(0, index);
}

const unsigned char* VisualFrame::getLidarDescriptor(const std::size_t index) const {
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_LIDAR_DESCRIPTORS_Data(channels_);
  CHECK_LT(static_cast<int>(index), descriptors.cols());
  return &descriptors.coeffRef(0, index);
}

double VisualFrame::getLidarKeypoint2DMeasurementUncertainty(const std::size_t index) const {
  Eigen::VectorXd& data =
      aslam::channels::get_LIDAR_KEYPOINT_2D_MEASUREMENT_UNCERTAINTIES_Data(
          channels_);
  CHECK_LT(static_cast<int>(index), data.rows());
  return data.coeff(index, 0);
}

void VisualFrame::setLidarTrackIds(const Eigen::VectorXi& track_ids_new) {
  if (!aslam::channels::has_LIDAR_TRACK_IDS_Channel(channels_)) {
    aslam::channels::add_LIDAR_TRACK_IDS_Channel(&channels_);
  }
  Eigen::VectorXi& data = aslam::channels::get_LIDAR_TRACK_IDS_Data(channels_);
  data = track_ids_new;
}

void VisualFrame::setLidarKeypoint3DMeasurements(
    const Eigen::Matrix3Xd& vectors_new) {
  if (!aslam::channels::has_LIDAR_3D_MEASUREMENTS_Channel(channels_)) {
    aslam::channels::add_LIDAR_3D_MEASUREMENTS_Channel(&channels_);
  }
  Eigen::Matrix3Xd& vectors =
      aslam::channels::get_LIDAR_3D_MEASUREMENTS_Data(channels_);
  vectors = vectors_new;
}

void VisualFrame::setLidarKeypoint2DMeasurements(
    const Eigen::Matrix2Xd& vectors_new) {
  if (!aslam::channels::has_LIDAR_2D_MEASUREMENTS_Channel(channels_)) {
    aslam::channels::add_LIDAR_2D_MEASUREMENTS_Channel(&channels_);
  }
  Eigen::Matrix2Xd& vectors =
      aslam::channels::get_LIDAR_2D_MEASUREMENTS_Data(channels_);
  vectors = vectors_new;
}

void VisualFrame::setLidarDescriptors(const DescriptorsT& descriptors_new) {
  if (!aslam::channels::has_LIDAR_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_LIDAR_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_LIDAR_DESCRIPTORS_Data(channels_);
  descriptors = descriptors_new;
}

void VisualFrame::setLidarDescriptors(
    const Eigen::Map<const DescriptorsT>& descriptors_new) {
  if (!aslam::channels::has_LIDAR_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_LIDAR_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_LIDAR_DESCRIPTORS_Data(channels_);
  descriptors = descriptors_new;
}

void VisualFrame::setLidarKeypoint2DMeasurementUncertainties(
    const Eigen::VectorXd& uncertainties_new) {
  if (!aslam::channels::has_LIDAR_KEYPOINT_2D_MEASUREMENT_UNCERTAINTIES_Channel(
          channels_)) {
    aslam::channels::add_LIDAR_KEYPOINT_2D_MEASUREMENT_UNCERTAINTIES_Channel(
        &channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_LIDAR_KEYPOINT_2D_MEASUREMENT_UNCERTAINTIES_Data(
          channels_);
  data = uncertainties_new;
}

void VisualFrame::swapLidarTrackIds(Eigen::VectorXi* track_ids_new) {
  if (!aslam::channels::has_LIDAR_TRACK_IDS_Channel(channels_)) {
    aslam::channels::add_LIDAR_TRACK_IDS_Channel(&channels_);
  }
  Eigen::VectorXi& track_ids =
      aslam::channels::get_LIDAR_TRACK_IDS_Data(channels_);
  track_ids.swap(*track_ids_new);
}

void VisualFrame::swapLidarKeypoint3DMeasurements(
    Eigen::Matrix3Xd* vectors_new) {
  if (!aslam::channels::has_LIDAR_3D_MEASUREMENTS_Channel(channels_)) {
    aslam::channels::add_LIDAR_3D_MEASUREMENTS_Channel(&channels_);
  }
  Eigen::Matrix3Xd& vectors =
      aslam::channels::get_LIDAR_3D_MEASUREMENTS_Data(channels_);
  vectors.swap(*vectors_new);
}

void VisualFrame::swapLidarKeypoint2DMeasurements(
    Eigen::Matrix2Xd* vectors_new) {
  if (!aslam::channels::has_LIDAR_2D_MEASUREMENTS_Channel(channels_)) {
    aslam::channels::add_LIDAR_2D_MEASUREMENTS_Channel(&channels_);
  }
  Eigen::Matrix2Xd& vectors =
      aslam::channels::get_LIDAR_2D_MEASUREMENTS_Data(channels_);
  vectors.swap(*vectors_new);
}

void VisualFrame::swapLidarDescriptors(DescriptorsT* descriptors_new) {
  if (!aslam::channels::has_LIDAR_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_LIDAR_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_LIDAR_DESCRIPTORS_Data(channels_);
  descriptors.swap(*descriptors_new);
}

void VisualFrame::swapLidarKeypoint2DMeasurementUncertainties(
    Eigen::VectorXd* uncertainties_new) {
  if (!aslam::channels::has_LIDAR_KEYPOINT_2D_MEASUREMENT_UNCERTAINTIES_Channel(
          channels_)) {
    aslam::channels::add_LIDAR_KEYPOINT_2D_MEASUREMENT_UNCERTAINTIES_Channel(
        &channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_LIDAR_KEYPOINT_2D_MEASUREMENT_UNCERTAINTIES_Data(
          channels_);
  data.swap(*uncertainties_new);
}

void VisualFrame::discardUntrackedLidarObservations(
    std::vector<size_t>* discarded_indices) {
  CHECK_NOTNULL(discarded_indices)->clear();
  CHECK(hasLidarTrackIds());
  const Eigen::VectorXi& track_ids = getLidarTrackIds();
  const std::size_t original_count = track_ids.rows();
  discarded_indices->reserve(original_count);

  for (std::size_t i = 0u; i < original_count; ++i) {
    if (track_ids(i) < 0) {
      discarded_indices->emplace_back(i);
    }
  }
  if (discarded_indices->empty()) {
    return;
  }
  if (hasLidarKeypoint3DMeasurements()) {
    common::stl_helpers::eraseIndicesFromContainer(
        *discarded_indices, original_count,
        getLidarKeypoint3DMeasurementsMutable());
  }
  if (hasLidarKeypoint2DMeasurements()) {
    common::stl_helpers::eraseIndicesFromContainer(
        *discarded_indices, original_count,
        getLidarKeypoint2DMeasurementsMutable());
  }
  if (hasLidarDescriptors()) {
    common::stl_helpers::OneDimensionAdapter<
        unsigned char, common::stl_helpers::kColumns>
        adapter(getLidarDescriptorsMutable());
    common::stl_helpers::eraseIndicesFromContainer(
        *discarded_indices, original_count, &adapter);
  }
  if (hasLidarKeypoint2DMeasurementUncertainties()) {
    common::stl_helpers::eraseIndicesFromContainer(
        *discarded_indices, original_count,
        getLidarKeypoint2DMeasurementUncertaintiesMutable());
  }
  common::stl_helpers::eraseIndicesFromContainer(
      *discarded_indices, original_count, getLidarTrackIdsMutable());
}
}  // namespace aslam
