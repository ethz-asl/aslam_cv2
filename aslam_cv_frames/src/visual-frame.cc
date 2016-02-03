#include "aslam/frames/visual-frame.h"

#include <memory>
#include <aslam/common/channel-definitions.h>
#include <aslam/common/stl-helpers.h>
#include <aslam/common/time.h>

namespace aslam {
VisualFrame::VisualFrame()
    : timestamp_nanoseconds_(time::getInvalidTime()),
      is_valid_(true) {}

VisualFrame::VisualFrame(const VisualFrame& other) {
  *this = other;
}

VisualFrame& VisualFrame::operator=(const VisualFrame& other) {
  timestamp_nanoseconds_ = other.timestamp_nanoseconds_;
  id_ = other.id_;
  camera_geometry_ = other.camera_geometry_;
  raw_camera_geometry_ = other.raw_camera_geometry_;

  channels_ = channels::cloneChannelGroup(other.channels_);
  is_valid_ = other.is_valid_;
  return *this;
}

bool VisualFrame::operator==(const VisualFrame& other) const {
  bool same = true;
  same &= timestamp_nanoseconds_ == other.timestamp_nanoseconds_;
  same &= channels::isChannelGroupEqual(channels_, other.channels_);
  same &= static_cast<bool>(camera_geometry_) ==
      static_cast<bool>(other.camera_geometry_);
  if (static_cast<bool>(camera_geometry_) &&
      static_cast<bool>(other.camera_geometry_)) {
    same &= (*camera_geometry_) == (*other.camera_geometry_);
  }
  same &= is_valid_ == other.is_valid_;
  return same;
}

bool VisualFrame::compareWithoutCameraGeometry(const VisualFrame& other) const {
  bool same = true;
  same &= timestamp_nanoseconds_ == other.timestamp_nanoseconds_;
  same &= channels::isChannelGroupEqual(channels_, other.channels_);
  same &= is_valid_ == other.is_valid_;
  return same;
}

bool VisualFrame::hasKeypointMeasurements() const {
  return aslam::channels::has_VISUAL_KEYPOINT_MEASUREMENTS_Channel(channels_);
}
bool VisualFrame::hasKeypointMeasurementUncertainties() const{
  return aslam::channels::has_VISUAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Channel(channels_);
}
bool VisualFrame::hasKeypointOrientations() const{
  return aslam::channels::has_VISUAL_KEYPOINT_ORIENTATIONS_Channel(channels_);
}
bool VisualFrame::hasKeypointScores() const {
  return aslam::channels::has_VISUAL_KEYPOINT_SCORES_Channel(channels_);
}
bool VisualFrame::hasKeypointScales() const{
  return aslam::channels::has_VISUAL_KEYPOINT_SCALES_Channel(channels_);
}
bool VisualFrame::hasDescriptors() const{
  return aslam::channels::has_DESCRIPTORS_Channel(channels_);
}
bool VisualFrame::hasTrackIds() const {
  return aslam::channels::has_TRACK_IDS_Channel(channels_);
}
bool VisualFrame::hasRawImage() const {
  return aslam::channels::has_RAW_IMAGE_Channel(channels_);
}

const Eigen::Matrix2Xd& VisualFrame::getKeypointMeasurements() const {
  return aslam::channels::get_VISUAL_KEYPOINT_MEASUREMENTS_Data(channels_);
}
const Eigen::VectorXd& VisualFrame::getKeypointMeasurementUncertainties() const {
  return aslam::channels::get_VISUAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Data(channels_);
}
const Eigen::VectorXd& VisualFrame::getKeypointScales() const {
  return aslam::channels::get_VISUAL_KEYPOINT_SCALES_Data(channels_);
}
const Eigen::VectorXd& VisualFrame::getKeypointOrientations() const {
  return aslam::channels::get_VISUAL_KEYPOINT_ORIENTATIONS_Data(channels_);
}
const Eigen::VectorXd& VisualFrame::getKeypointScores() const {
  return aslam::channels::get_VISUAL_KEYPOINT_SCORES_Data(channels_);
}
const VisualFrame::DescriptorsT& VisualFrame::getDescriptors() const {
  return aslam::channels::get_DESCRIPTORS_Data(channels_);
}
const Eigen::VectorXi& VisualFrame::getTrackIds() const {
  return aslam::channels::get_TRACK_IDS_Data(channels_);
}
const cv::Mat& VisualFrame::getRawImage() const {
  return aslam::channels::get_RAW_IMAGE_Data(channels_);
}

void VisualFrame::releaseRawImage() {
  aslam::channels::remove_RAW_IMAGE_Channel(&channels_);
}

Eigen::Matrix2Xd* VisualFrame::getKeypointMeasurementsMutable() {
  Eigen::Matrix2Xd& keypoints =
      aslam::channels::get_VISUAL_KEYPOINT_MEASUREMENTS_Data(channels_);
    return &keypoints;
}
Eigen::VectorXd* VisualFrame::getKeypointMeasurementUncertaintiesMutable() {
  Eigen::VectorXd& uncertainties =
      aslam::channels::get_VISUAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Data(channels_);
    return &uncertainties;
}
Eigen::VectorXd* VisualFrame::getKeypointScalesMutable() {
  Eigen::VectorXd& scales =
      aslam::channels::get_VISUAL_KEYPOINT_SCALES_Data(channels_);
    return &scales;
}
Eigen::VectorXd* VisualFrame::getKeypointOrientationsMutable() {
  Eigen::VectorXd& orientations =
      aslam::channels::get_VISUAL_KEYPOINT_ORIENTATIONS_Data(channels_);
    return &orientations;
}
Eigen::VectorXd* VisualFrame::getKeypointScoresMutable() {
  Eigen::VectorXd& scores =
      aslam::channels::get_VISUAL_KEYPOINT_SCORES_Data(channels_);
    return &scores;
}
VisualFrame::DescriptorsT* VisualFrame::getDescriptorsMutable() {
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_DESCRIPTORS_Data(channels_);
  return &descriptors;
}
Eigen::VectorXi* VisualFrame::getTrackIdsMutable() {
  Eigen::VectorXi& track_ids =
      aslam::channels::get_TRACK_IDS_Data(channels_);
  return &track_ids;
}
cv::Mat* VisualFrame::getRawImageMutable() {
  cv::Mat& image =
      aslam::channels::get_RAW_IMAGE_Data(channels_);
  return &image;
}

const Eigen::Block<Eigen::Matrix2Xd, 2, 1>
VisualFrame::getKeypointMeasurement(size_t index) const {
  Eigen::Matrix2Xd& keypoints =
      aslam::channels::get_VISUAL_KEYPOINT_MEASUREMENTS_Data(channels_);
  CHECK_LT(static_cast<int>(index), keypoints.cols());
  return keypoints.block<2, 1>(0, index);
}
double VisualFrame::getKeypointMeasurementUncertainty(size_t index) const {
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Data(channels_);
  CHECK_LT(static_cast<int>(index), data.rows());
  return data.coeff(index, 0);
}
double VisualFrame::getKeypointScale(size_t index) const {
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_SCALES_Data(channels_);
  CHECK_LT(static_cast<int>(index), data.rows());
  return data.coeff(index, 0);
}
double VisualFrame::getKeypointOrientation(size_t index) const {
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_ORIENTATIONS_Data(channels_);
  CHECK_LT(static_cast<int>(index), data.rows());
  return data.coeff(index, 0);
}
double VisualFrame::getKeypointScore(size_t index) const {
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_SCORES_Data(channels_);
  CHECK_LT(static_cast<int>(index), data.rows());
  return data.coeff(index, 0);
}
const unsigned char* VisualFrame::getDescriptor(size_t index) const {
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_DESCRIPTORS_Data(channels_);
  CHECK_LT(static_cast<int>(index), descriptors.cols());
  return &descriptors.coeffRef(0, index);
}
int VisualFrame::getTrackId(size_t index) const {
  Eigen::VectorXi& track_ids =
      aslam::channels::get_TRACK_IDS_Data(channels_);
  CHECK_LT(static_cast<int>(index), track_ids.rows());
  return track_ids.coeff(index, 0);
}

void VisualFrame::setKeypointMeasurements(
    const Eigen::Matrix2Xd& keypoints_new) {
  if (!aslam::channels::has_VISUAL_KEYPOINT_MEASUREMENTS_Channel(channels_)) {
    aslam::channels::add_VISUAL_KEYPOINT_MEASUREMENTS_Channel(&channels_);
  }
  Eigen::Matrix2Xd& keypoints =
      aslam::channels::get_VISUAL_KEYPOINT_MEASUREMENTS_Data(channels_);
  keypoints = keypoints_new;
}
void VisualFrame::setKeypointMeasurementUncertainties(
    const Eigen::VectorXd& uncertainties_new) {
  if (!aslam::channels::has_VISUAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Channel(channels_)) {
    aslam::channels::add_VISUAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Data(channels_);
  data = uncertainties_new;
}
void VisualFrame::setKeypointScales(
    const Eigen::VectorXd& scales_new) {
  if (!aslam::channels::has_VISUAL_KEYPOINT_SCALES_Channel(channels_)) {
    aslam::channels::add_VISUAL_KEYPOINT_SCALES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_SCALES_Data(channels_);
  data = scales_new;
}
void VisualFrame::setKeypointOrientations(
    const Eigen::VectorXd& orientations_new) {
  if (!aslam::channels::has_VISUAL_KEYPOINT_ORIENTATIONS_Channel(channels_)) {
    aslam::channels::add_VISUAL_KEYPOINT_ORIENTATIONS_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_ORIENTATIONS_Data(channels_);
  data = orientations_new;
}
void VisualFrame::setKeypointScores(
    const Eigen::VectorXd& scores_new) {
  if (!aslam::channels::has_VISUAL_KEYPOINT_SCORES_Channel(channels_)) {
    aslam::channels::add_VISUAL_KEYPOINT_SCORES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_SCORES_Data(channels_);
  data = scores_new;
}
void VisualFrame::setDescriptors(
    const DescriptorsT& descriptors_new) {
  if (!aslam::channels::has_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_DESCRIPTORS_Data(channels_);
  descriptors = descriptors_new;
}
void VisualFrame::setDescriptors(
    const Eigen::Map<const DescriptorsT>& descriptors_new) {
  if (!aslam::channels::has_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_DESCRIPTORS_Data(channels_);
  descriptors = descriptors_new;
}
void VisualFrame::setTrackIds(const Eigen::VectorXi& track_ids_new) {
  if (!aslam::channels::has_TRACK_IDS_Channel(channels_)) {
    aslam::channels::add_TRACK_IDS_Channel(&channels_);
  }
  Eigen::VectorXi& data =
      aslam::channels::get_TRACK_IDS_Data(channels_);
  data = track_ids_new;
}

void VisualFrame::setRawImage(const cv::Mat& image_new) {
  if (!aslam::channels::has_RAW_IMAGE_Channel(channels_)) {
    aslam::channels::add_RAW_IMAGE_Channel(&channels_);
  }
  cv::Mat& image =
      aslam::channels::get_RAW_IMAGE_Data(channels_);
  image = image_new;
}

void VisualFrame::swapKeypointMeasurements(Eigen::Matrix2Xd* keypoints_new) {
  if (!aslam::channels::has_VISUAL_KEYPOINT_MEASUREMENTS_Channel(channels_)) {
    aslam::channels::add_VISUAL_KEYPOINT_MEASUREMENTS_Channel(&channels_);
  }
  Eigen::Matrix2Xd& keypoints =
      aslam::channels::get_VISUAL_KEYPOINT_MEASUREMENTS_Data(channels_);
  keypoints.swap(*keypoints_new);
}
void VisualFrame::swapKeypointMeasurementUncertainties(Eigen::VectorXd* uncertainties_new) {
  if (!aslam::channels::has_VISUAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Channel(channels_)) {
    aslam::channels::add_VISUAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES_Data(channels_);
  data.swap(*uncertainties_new);
}
void VisualFrame::swapKeypointScales(Eigen::VectorXd* scales_new) {
  if (!aslam::channels::has_VISUAL_KEYPOINT_SCALES_Channel(channels_)) {
    aslam::channels::add_VISUAL_KEYPOINT_SCALES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_SCALES_Data(channels_);
  data.swap(*scales_new);
}
void VisualFrame::swapKeypointOrientations(Eigen::VectorXd* orientations_new) {
  if (!aslam::channels::has_VISUAL_KEYPOINT_ORIENTATIONS_Channel(channels_)) {
    aslam::channels::add_VISUAL_KEYPOINT_ORIENTATIONS_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_ORIENTATIONS_Data(channels_);
  data.swap(*orientations_new);
}
void VisualFrame::swapKeypointScores(Eigen::VectorXd* scores_new) {
  if (!aslam::channels::has_VISUAL_KEYPOINT_SCORES_Channel(channels_)) {
    aslam::channels::add_VISUAL_KEYPOINT_SCORES_Channel(&channels_);
  }
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_SCORES_Data(channels_);
  data.swap(*scores_new);
}
void VisualFrame::swapDescriptors(DescriptorsT* descriptors_new) {
  if (!aslam::channels::has_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_DESCRIPTORS_Data(channels_);
  descriptors.swap(*descriptors_new);
}

void VisualFrame::swapTrackIds(Eigen::VectorXi* track_ids_new) {
  if (!aslam::channels::has_TRACK_IDS_Channel(channels_)) {
    aslam::channels::add_TRACK_IDS_Channel(&channels_);
  }
  Eigen::VectorXi& track_ids = aslam::channels::get_TRACK_IDS_Data(channels_);
  track_ids.swap(*track_ids_new);
}

void VisualFrame::clearKeypointChannels() {
  Eigen::Matrix2Xd zero_keypoints;
  setKeypointMeasurements(zero_keypoints);

  Eigen::VectorXi zero_vector_int = Eigen::VectorXi::Zero(zero_keypoints.cols());
  setTrackIds(zero_vector_int);

  Eigen::VectorXd zero_vector_double = Eigen::VectorXd::Zero(zero_keypoints.cols());
  setKeypointMeasurementUncertainties(zero_vector_double);
  setKeypointOrientations(zero_vector_double);
  setKeypointScores(zero_vector_double);
  setKeypointScales(zero_vector_double);
  setDescriptors(aslam::VisualFrame::DescriptorsT());
}

const Camera::ConstPtr VisualFrame::getCameraGeometry() const {
  return camera_geometry_;
}

void VisualFrame::setCameraGeometry(const Camera::ConstPtr& camera) {
  camera_geometry_ = camera;
}

const Camera::ConstPtr VisualFrame::getRawCameraGeometry() const {
  return raw_camera_geometry_;
}

void VisualFrame::setRawCameraGeometry(const Camera::ConstPtr& camera) {
  raw_camera_geometry_ = camera;
}

void VisualFrame::print(std::ostream& out, const std::string& label) const {
  if(label.size() > 0) {
    out << label << std::endl;
  }
  out << "VisualFrame(" << this->id_ << ")" << std::endl;
  out << "  timestamp:          " << this->timestamp_nanoseconds_ << std::endl;
  if(camera_geometry_) {
    camera_geometry_->printParameters(out, "  VisualFrame::camera");
  } else {
    out << "  VisualFrame::camera is NULL" << std::endl;
  }
  channels_.printParameters(out);
}

aslam::ProjectionResult VisualFrame::getKeypointInRawImageCoordinates(
    size_t keypoint_idx, Eigen::Vector2d* keypoint_raw_coordinates) const {
  CHECK_NOTNULL(keypoint_raw_coordinates);
  const Eigen::Vector2d& keypoint = getKeypointMeasurement(keypoint_idx);
  return toRawImageCoordinates(keypoint, keypoint_raw_coordinates);
}

aslam::ProjectionResult VisualFrame::toRawImageCoordinates(
    const Eigen::Vector2d& keypoint, Eigen::Vector2d* out_image_coordinates) const {
  CHECK_NOTNULL(out_image_coordinates);
  Eigen::Vector3d bearing;
  // Creating a bearing vector from the transformed camera, then projecting this
  // bearing should recover the raw image coordinates.
  bool success = camera_geometry_->backProject3(keypoint, &bearing);
  if(success) {
    return raw_camera_geometry_->project3(bearing, out_image_coordinates );
  } else {
    return ProjectionResult::PROJECTION_INVALID;
  }
}

void VisualFrame::toRawImageCoordinatesVectorized(
    const Eigen::Matrix2Xd& keypoints, Eigen::Matrix2Xd* out_image_coordinates,
    std::vector<aslam::ProjectionResult>* results) const {
  CHECK_NOTNULL(out_image_coordinates);
  CHECK_NOTNULL(results);
  Eigen::Matrix3Xd bearings;
  std::vector<unsigned char> success;
  camera_geometry_->backProject3Vectorized(keypoints, &bearings, &success);
  raw_camera_geometry_->project3Vectorized(bearings, out_image_coordinates, results);
  for(size_t i = 0; i < success.size(); ++i) {
    if(!success[i]){
      (*results)[i] = ProjectionResult::PROJECTION_INVALID;
    }
  }
}

size_t VisualFrame::getDescriptorSizeBytes() const {
  return getDescriptors().rows() * sizeof(DescriptorsT::Scalar);
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
}

VisualFrame::Ptr VisualFrame::createEmptyTestVisualFrame(const aslam::Camera::ConstPtr& camera,
                                                         int64_t timestamp_nanoseconds) {
  aslam::VisualFrame::Ptr frame(new aslam::VisualFrame);
  frame->setCameraGeometry(camera);
  frame->setTimestampNanoseconds(timestamp_nanoseconds);
  Eigen::Matrix2Xd keypoint_measurements = Eigen::Matrix2Xd::Zero(2, 0);
  frame->swapKeypointMeasurements(&keypoint_measurements);
  Eigen::VectorXd keypoint_uncertainties = Eigen::VectorXd::Zero(0);
  frame->swapKeypointMeasurementUncertainties(&keypoint_uncertainties);
  aslam::VisualFrame::DescriptorsT descriptors = aslam::VisualFrame::DescriptorsT::Zero(48, 0);
  frame->swapDescriptors(&descriptors);
  aslam::FrameId id;
  id.randomize();
  frame->setId(id);
  return frame;
}

void VisualFrame::discardUntrackedObservations(
    std::vector<size_t>* discarded_indices) {
  CHECK_NOTNULL(discarded_indices)->clear();
  CHECK(hasTrackIds());
  const Eigen::VectorXi& track_ids = getTrackIds();
  const int original_count = track_ids.rows();
  discarded_indices->reserve(original_count);
  for (int i = 0; i < original_count; ++i) {
    if (track_ids(i) < 0) {
      discarded_indices->emplace_back(i);
    }
  }
  if (discarded_indices->empty()) {
    return;
  }

  if (hasKeypointMeasurements()) {
    common::stl_helpers::eraseIndicesFromContainer(
        *discarded_indices, original_count, getKeypointMeasurementsMutable());
  }
  if (hasKeypointMeasurementUncertainties()) {
    common::stl_helpers::eraseIndicesFromContainer(
        *discarded_indices, original_count,
        getKeypointMeasurementUncertaintiesMutable());
  }
  if (hasKeypointOrientations()) {
    common::stl_helpers::eraseIndicesFromContainer(
        *discarded_indices, original_count, getKeypointOrientationsMutable());
  }
  if (hasKeypointScores()) {
    common::stl_helpers::eraseIndicesFromContainer(
        *discarded_indices, original_count, getKeypointScoresMutable());
  }
  if (hasKeypointScales()) {
    common::stl_helpers::eraseIndicesFromContainer(
        *discarded_indices, original_count, getKeypointScalesMutable());
  }
  if (hasDescriptors()) {
    common::stl_helpers::OneDimensionAdapter<unsigned char,
    common::stl_helpers::kColumns> adapter(getDescriptorsMutable());
    common::stl_helpers::eraseIndicesFromContainer(
        *discarded_indices, original_count, &adapter);
  }
  common::stl_helpers::eraseIndicesFromContainer(
      *discarded_indices, original_count, getTrackIdsMutable());
}

}  // namespace aslam
