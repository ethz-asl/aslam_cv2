#include "aslam/frames/visual-frame.h"

#include <memory>
#include <aslam/common/channel-definitions-external-features.h>
#include <aslam/common/stl-helpers.h>
#include <aslam/common/time.h>

namespace aslam {

bool VisualFrame::hasExternalKeypointMeasurements() const {
  return aslam::channels::has_EXTERNAL_KEYPOINT_MEASUREMENTS_Channel(channels_);
}
bool VisualFrame::hasExternalKeypointMeasurementUncertainties() const{
  return aslam::channels::has_EXTERNAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Channel(channels_);
}
bool VisualFrame::hasExternalKeypointOrientations() const{
  return aslam::channels::has_EXTERNAL_KEYPOINT_ORIENTATIONS_Channel(channels_);
}
bool VisualFrame::hasExternalKeypointScores() const {
  return aslam::channels::has_EXTERNAL_KEYPOINT_SCORES_Channel(channels_);
}
bool VisualFrame::hasExternalKeypointScales() const{
  return aslam::channels::has_EXTERNAL_KEYPOINT_SCALES_Channel(channels_);
}
bool VisualFrame::hasExternalDescriptors() const{
  return aslam::channels::has_EXTERNAL_DESCRIPTORS_Channel(channels_);
}
bool VisualFrame::hasExternalTrackIds() const {
  return aslam::channels::has_EXTERNAL_TRACK_IDS_Channel(channels_);
}

const Eigen::Matrix2Xd& VisualFrame::getExternalKeypointMeasurements() const {
  return aslam::channels::get_EXTERNAL_KEYPOINT_MEASUREMENTS_Data(channels_);
}
const Eigen::VectorXd& VisualFrame::getExternalKeypointMeasurementUncertainties() const {
  return aslam::channels::get_EXTERNAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Data(channels_);
}
const Eigen::VectorXd& VisualFrame::getExternalKeypointScales() const {
  return aslam::channels::get_EXTERNAL_KEYPOINT_SCALES_Data(channels_);
}
const Eigen::VectorXd& VisualFrame::getExternalKeypointOrientations() const {
  return aslam::channels::get_EXTERNAL_KEYPOINT_ORIENTATIONS_Data(channels_);
}
const Eigen::VectorXd& VisualFrame::getExternalKeypointScores() const {
  return aslam::channels::get_EXTERNAL_KEYPOINT_SCORES_Data(channels_);
}
const VisualFrame::DescriptorsT& VisualFrame::getExternalDescriptors() const {
  return aslam::channels::get_EXTERNAL_DESCRIPTORS_Data(channels_);
}
const Eigen::VectorXi& VisualFrame::getExternalTrackIds() const {
  return aslam::channels::get_EXTERNAL_TRACK_IDS_Data(channels_);
}

Eigen::Matrix2Xd* VisualFrame::getExternalKeypointMeasurementsMutable() {
  Eigen::Matrix2Xd& keypoints =
      aslam::channels::get_EXTERNAL_KEYPOINT_MEASUREMENTS_Data(channels_);
    return &keypoints;
}
Eigen::VectorXd* VisualFrame::getExternalKeypointMeasurementUncertaintiesMutable() {
  Eigen::VectorXd& uncertainties =
      aslam::channels::get_EXTERNAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Data(channels_);
    return &uncertainties;
}
Eigen::VectorXd* VisualFrame::getExternalKeypointScalesMutable() {
  Eigen::VectorXd& scales =
      aslam::channels::get_EXTERNAL_KEYPOINT_SCALES_Data(channels_);
    return &scales;
}
Eigen::VectorXd* VisualFrame::getExternalKeypointOrientationsMutable() {
  Eigen::VectorXd& orientations =
      aslam::channels::get_EXTERNAL_KEYPOINT_ORIENTATIONS_Data(channels_);
    return &orientations;
}
Eigen::VectorXd* VisualFrame::getExternalKeypointScoresMutable() {
  Eigen::VectorXd& scores =
      aslam::channels::get_EXTERNAL_KEYPOINT_SCORES_Data(channels_);
    return &scores;
}
VisualFrame::DescriptorsT* VisualFrame::getExternalDescriptorsMutable() {
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_EXTERNAL_DESCRIPTORS_Data(channels_);
  return &descriptors;
}
Eigen::VectorXi* VisualFrame::getExternalTrackIdsMutable() {
  Eigen::VectorXi& track_ids =
      aslam::channels::get_EXTERNAL_TRACK_IDS_Data(channels_);
  return &track_ids;
}

const Eigen::Block<Eigen::Matrix2Xd, 2, 1>
VisualFrame::getExternalKeypointMeasurement(size_t index) const {
  Eigen::Matrix2Xd& keypoints =
      aslam::channels::get_EXTERNAL_KEYPOINT_MEASUREMENTS_Data(channels_);
  CHECK_LT(static_cast<int>(index), keypoints.cols());
  return keypoints.block<2, 1>(0, index);
}
double VisualFrame::getExternalKeypointMeasurementUncertainty(size_t index) const {
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Data(channels_);
  CHECK_LT(static_cast<int>(index), data.rows());
  return data.coeff(index, 0);
}
double VisualFrame::getExternalKeypointScale(size_t index) const {
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_SCALES_Data(channels_);
  CHECK_LT(static_cast<int>(index), data.rows());
  return data.coeff(index, 0);
}
double VisualFrame::getExternalKeypointOrientation(size_t index) const {
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_ORIENTATIONS_Data(channels_);
  CHECK_LT(static_cast<int>(index), data.rows());
  return data.coeff(index, 0);
}
double VisualFrame::getExternalKeypointScore(size_t index) const {
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_SCORES_Data(channels_);
  CHECK_LT(static_cast<int>(index), data.rows());
  return data.coeff(index, 0);
}
/*const unsigned char* VisualFrame::getExternalDescriptor(size_t index) const {
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_EXTERNAL_DESCRIPTORS_Data(channels_);
  CHECK_LT(static_cast<int>(index), descriptors.cols());
  return &descriptors.coeffRef(0, index);
}*/
int VisualFrame::getExternalTrackId(size_t index) const {
  Eigen::VectorXi& track_ids =
      aslam::channels::get_EXTERNAL_TRACK_IDS_Data(channels_);
  CHECK_LT(static_cast<int>(index), track_ids.rows());
  return track_ids.coeff(index, 0);
}

void VisualFrame::setExternalKeypointMeasurements(
    const Eigen::Matrix2Xd& keypoints_new) {
  if (!aslam::channels::has_EXTERNAL_KEYPOINT_MEASUREMENTS_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_KEYPOINT_MEASUREMENTS_Channel(&channels_);
  }
  Eigen::Matrix2Xd& keypoints =
      aslam::channels::get_EXTERNAL_KEYPOINT_MEASUREMENTS_Data(channels_);
  keypoints = keypoints_new;
}
void VisualFrame::setExternalKeypointMeasurementUncertainties(
    const Eigen::VectorXd& uncertainties_new) {
  if (!aslam::channels::has_EXTERNAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Data(channels_);
  data = uncertainties_new;
}
void VisualFrame::setExternalKeypointScales(
    const Eigen::VectorXd& scales_new) {
  if (!aslam::channels::has_EXTERNAL_KEYPOINT_SCALES_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_KEYPOINT_SCALES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_SCALES_Data(channels_);
  data = scales_new;
}
void VisualFrame::setExternalKeypointOrientations(
    const Eigen::VectorXd& orientations_new) {
  if (!aslam::channels::has_EXTERNAL_KEYPOINT_ORIENTATIONS_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_KEYPOINT_ORIENTATIONS_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_ORIENTATIONS_Data(channels_);
  data = orientations_new;
}
void VisualFrame::setExternalKeypointScores(
    const Eigen::VectorXd& scores_new) {
  if (!aslam::channels::has_EXTERNAL_KEYPOINT_SCORES_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_KEYPOINT_SCORES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_SCORES_Data(channels_);
  data = scores_new;
}
void VisualFrame::setExternalDescriptors(
    const DescriptorsT& descriptors_new) {
  if (!aslam::channels::has_EXTERNAL_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_EXTERNAL_DESCRIPTORS_Data(channels_);
  descriptors = descriptors_new;
}
void VisualFrame::setExternalDescriptors(
    const Eigen::Map<const DescriptorsT>& descriptors_new) {
  if (!aslam::channels::has_EXTERNAL_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_EXTERNAL_DESCRIPTORS_Data(channels_);
  descriptors = descriptors_new;
}
void VisualFrame::setExternalTrackIds(const Eigen::VectorXi& track_ids_new) {
  if (!aslam::channels::has_EXTERNAL_TRACK_IDS_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_TRACK_IDS_Channel(&channels_);
  }
  Eigen::VectorXi& data =
      aslam::channels::get_EXTERNAL_TRACK_IDS_Data(channels_);
  data = track_ids_new;
}

void VisualFrame::swapExternalKeypointMeasurements(Eigen::Matrix2Xd* keypoints_new) {
  if (!aslam::channels::has_EXTERNAL_KEYPOINT_MEASUREMENTS_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_KEYPOINT_MEASUREMENTS_Channel(&channels_);
  }
  Eigen::Matrix2Xd& keypoints =
      aslam::channels::get_EXTERNAL_KEYPOINT_MEASUREMENTS_Data(channels_);
  keypoints.swap(*keypoints_new);
}
void VisualFrame::swapExternalKeypointMeasurementUncertainties(Eigen::VectorXd* uncertainties_new) {
  if (!aslam::channels::has_EXTERNAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Data(channels_);
  data.swap(*uncertainties_new);
}
void VisualFrame::swapExternalKeypointScales(Eigen::VectorXd* scales_new) {
  if (!aslam::channels::has_EXTERNAL_KEYPOINT_SCALES_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_KEYPOINT_SCALES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_SCALES_Data(channels_);
  data.swap(*scales_new);
}
void VisualFrame::swapExternalKeypointOrientations(Eigen::VectorXd* orientations_new) {
  if (!aslam::channels::has_EXTERNAL_KEYPOINT_ORIENTATIONS_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_KEYPOINT_ORIENTATIONS_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_ORIENTATIONS_Data(channels_);
  data.swap(*orientations_new);
}
void VisualFrame::swapExternalKeypointScores(Eigen::VectorXd* scores_new) {
  if (!aslam::channels::has_EXTERNAL_KEYPOINT_SCORES_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_KEYPOINT_SCORES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_EXTERNAL_KEYPOINT_SCORES_Data(channels_);
  data.swap(*scores_new);
}
void VisualFrame::swapExternalDescriptors(DescriptorsT* descriptors_new) {
  if (!aslam::channels::has_EXTERNAL_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_EXTERNAL_DESCRIPTORS_Data(channels_);
  descriptors.swap(*descriptors_new);
}
void VisualFrame::swapExternalTrackIds(Eigen::VectorXi* track_ids_new) {
  if (!aslam::channels::has_EXTERNAL_TRACK_IDS_Channel(channels_)) {
    aslam::channels::add_EXTERNAL_TRACK_IDS_Channel(&channels_);
  }
  Eigen::VectorXi& track_ids = aslam::channels::get_EXTERNAL_TRACK_IDS_Data(channels_);
  track_ids.swap(*track_ids_new);
}

void VisualFrame::clearExternalKeypointChannels() {
  Eigen::Matrix2Xd zero_keypoints;
  setExternalKeypointMeasurements(zero_keypoints);

  Eigen::VectorXi zero_vector_int = Eigen::VectorXi::Zero(zero_keypoints.cols());
  setExternalTrackIds(zero_vector_int);

  Eigen::VectorXd zero_vector_double = Eigen::VectorXd::Zero(zero_keypoints.cols());
  setExternalKeypointMeasurementUncertainties(zero_vector_double);
  setExternalKeypointOrientations(zero_vector_double);
  setExternalKeypointScores(zero_vector_double);
  setExternalKeypointScales(zero_vector_double);
  setExternalDescriptors(aslam::VisualFrame::DescriptorsT());
}


/*size_t VisualFrame::getExternalDescriptorSizeBytes() const {
  return getExternalDescriptors().rows() * sizeof(DescriptorsT::Scalar);
}

Eigen::Matrix3Xd VisualFrame::getNormalizedBearingVectors(
    const std::vector<size_t>& keypoint_indices,
    std::vector<unsigned char>* backprojection_success) const {
  CHECK_NOTNULL(backprojection_success);
  if (keypoint_indices.empty()) {
    backprojection_success->clear();
    return Eigen::Matrix3Xd(3, 0);
  }

  const aslam::Camera& camera = *CHECK_NOTNULL(getCameraGeometry().get());
  const Eigen::Matrix2Xd& keypoints = getKeypointMeasurements();
  const size_t num_keypoints = getNumKeypointMeasurements();

  Eigen::Matrix2Xd keypoints_reduced;
  keypoints_reduced.resize(Eigen::NoChange, keypoint_indices.size());

  size_t list_idx = 0;
  for (const size_t keypoint_idx : keypoint_indices) {
    CHECK_LE(keypoint_idx, num_keypoints);
    keypoints_reduced.col(list_idx++) = keypoints.col(keypoint_idx);
  }

  Eigen::Matrix3Xd points_3d;
  camera.backProject3Vectorized(keypoints_reduced, &points_3d, backprojection_success);
  return points_3d.colwise().normalized();
}*/

}  // namespace aslam
