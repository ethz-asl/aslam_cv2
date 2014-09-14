#include <memory>
#include <aslam/frames/visual-frame.h>
#include <aslam/common/channel-definitions.h>
#include <aslam/common/time.h>

namespace aslam {
VisualFrame::VisualFrame() :
    stamp_(getInvalidTime()), hardwareStamp_(getInvalidTime()), systemStamp_(getInvalidTime()) {}

VisualFrame::~VisualFrame(){}

bool VisualFrame::operator==(const VisualFrame& other) const {
  bool same = true;
  // TODO(slynen): Better iterate over channels and compare data instead of pointers.
  same &= stamp_ == other.stamp_;
  same &= hardwareStamp_ == other.hardwareStamp_;
  same &= systemStamp_ == other.systemStamp_;
  same &= channels_ == other.channels_;
  same &= static_cast<bool>(camera_geometry_) ==
      static_cast<bool>(other.camera_geometry_);
  if (static_cast<bool>(camera_geometry_) &&
      static_cast<bool>(other.camera_geometry_)) {
    same &= (*camera_geometry_) == (*other.camera_geometry_);
  }
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
bool VisualFrame::hasKeypointScales() const{
  return aslam::channels::has_VISUAL_KEYPOINT_SCALES_Channel(channels_);
}
bool VisualFrame::hasBriskDescriptors() const{
  return aslam::channels::has_BRISK_DESCRIPTORS_Channel(channels_);
}
bool VisualFrame::hasRawImage() const {
  return aslam::channels::has_IMAGE_Channel(channels_);
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
const VisualFrame::DescriptorsT& VisualFrame::getBriskDescriptors() const {
  return aslam::channels::get_BRISK_DESCRIPTORS_Data(channels_);
}
const cv::Mat& VisualFrame::getRawImage() const {
  return aslam::channels::get_IMAGE_Data(channels_);
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
VisualFrame::DescriptorsT* VisualFrame::getBriskDescriptorsMutable() {
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_BRISK_DESCRIPTORS_Data(channels_);
  return &descriptors;
}
cv::Mat* VisualFrame::getRawImageMutable() {
  cv::Mat& image =
      aslam::channels::get_IMAGE_Data(channels_);
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
  CHECK_LT(static_cast<int>(index), data.cols());
  return data.coeff(0, index);
}
double VisualFrame::getKeypointScale(size_t index) const {
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_SCALES_Data(channels_);
  CHECK_LT(static_cast<int>(index), data.cols());
  return data.coeff(0, index);
}
double VisualFrame::getKeypointOrientation(size_t index) const {
  Eigen::VectorXd& data =
      aslam::channels::get_VISUAL_KEYPOINT_ORIENTATIONS_Data(channels_);
  CHECK_LT(static_cast<int>(index), data.cols());
  return data.coeff(0, index);
}
const unsigned char* VisualFrame::getBriskDescriptor(size_t index) const {
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_BRISK_DESCRIPTORS_Data(channels_);
  CHECK_LT(static_cast<int>(index), descriptors.cols());
  return &descriptors.coeffRef(0, index);
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
void VisualFrame::setBriskDescriptors(
    const DescriptorsT& descriptors_new) {
  if (!aslam::channels::has_BRISK_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_BRISK_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_BRISK_DESCRIPTORS_Data(channels_);
  descriptors = descriptors_new;
}
void VisualFrame::setRawImage(const cv::Mat& image_new) {
  if (!aslam::channels::has_IMAGE_Channel(channels_)) {
    aslam::channels::add_IMAGE_Channel(&channels_);
  }
  cv::Mat& image =
      aslam::channels::get_IMAGE_Data(channels_);
  image = image_new;
}

const Camera::ConstPtr VisualFrame::getCameraGeometry() const {
  return camera_geometry_;
}

void VisualFrame::setCameraGeometry(const Camera::Ptr& camera) {
  camera_geometry_ = camera;
}

void VisualFrame::print(std::ostream& out, const std::string& label) const {
  if(label.size() > 0) {
    out << label << std::endl;
  }
  out << "VisualFrame(" << this->id_ << ")" << std::endl;
  out << "  timestamp:          " << this->stamp_ << std::endl;
  out << "  system timestamp:   " << this->systemStamp_ << std::endl;
  out << "  hardware timestamp: " << this->hardwareStamp_ << std::endl;
  if(camera_geometry_) {
    camera_geometry_->printParameters(out, "  VisualFrame::camera");
  } else {
    out << "  VisualFrame::camera is NULL" << std::endl;
  }
  if(! channels_.empty()) {
    out << "  Channels:" << std::endl;
    aslam::channels::ChannelGroup::const_iterator it = channels_.begin();
    for( ; it != channels_.end(); ++it) {
      out << "   - " << it->first << std::endl;
    }
  } else {
    out << "  Channels: empty" << std::endl;
  }
}

aslam::ProjectionResult VisualFrame::toRawImageCoordinates(const Eigen::Vector2d& keypoint,
                                                           Eigen::Vector2d* out_image_coordinates) {
  CHECK_NOTNULL(out_image_coordinates);
  Eigen::Vector3d bearing;
  // Creating a bearing vector from the transformed camera, then projecting this
  // bearing should recover the raw image coordinates.
  bool success = camera_geometry_->backProject3( keypoint, &bearing );
  if(success) {
    return raw_camera_geometry_->project3(bearing, out_image_coordinates );
  } else {
    return ProjectionResult::Status::PROJECTION_INVALID;
  }
}

void VisualFrame::toRawImageCoordinatesVectorized(const Eigen::Matrix2Xd& keypoints,
                                                  Eigen::Matrix2Xd* out_image_coordinates,
                                                  std::vector<aslam::ProjectionResult>* results) {
  CHECK_NOTNULL(out_image_coordinates);
  CHECK_NOTNULL(results);
  Eigen::Matrix3Xd bearings;
  std::vector<bool> success;
  camera_geometry_->backProject3Vectorized(keypoints, &bearings, &success);
  raw_camera_geometry_->project3Vectorized(bearings, out_image_coordinates, results);
  for(size_t i = 0; i < success.size(); ++i) {
    if(!success[i]){
      (*results)[i] = ProjectionResult::Status::PROJECTION_INVALID;
    }
  }
}
}  // namespace aslam
