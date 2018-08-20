#include "aslam/pipeline/synchronous-visual-npipeline.h"

#include <aslam/cameras/camera.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/common/memory.h>
#include <aslam/common/thread-pool.h>
#include <aslam/common/time.h>
#include <aslam/frames/visual-nframe.h>
#include <aslam/pipeline/visual-pipeline.h>
#include <aslam/pipeline/visual-pipeline-null.h>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>

namespace aslam {

SynchronousVisualNPipeline::SynchronousVisualNPipeline(
    const std::vector<std::shared_ptr<VisualPipeline> >& pipelines,
    const std::shared_ptr<NCamera>& input_camera_system,
    const std::shared_ptr<NCamera>& output_camera_system,
    const int64_t timestamp_tolerance_ns) :
      pipelines_(pipelines),
      input_camera_system_(input_camera_system),
      output_camera_system_(output_camera_system),
      timestamp_tolerance_ns_(timestamp_tolerance_ns)  {
  // Defensive programming ninjitsu.
  CHECK_NOTNULL(input_camera_system_.get());
  CHECK_NOTNULL(output_camera_system.get());
  CHECK_GT(input_camera_system_->numCameras(), 0u);
  CHECK_EQ(input_camera_system_->numCameras(),
           output_camera_system_->numCameras());
  CHECK_EQ(input_camera_system_->numCameras(), pipelines.size());
  CHECK_GE(timestamp_tolerance_ns, 0);

  for (size_t i = 0u; i < pipelines.size(); ++i) {
    CHECK_NOTNULL(pipelines[i].get());
    // Check that the input cameras actually point to the same object.
    CHECK_EQ(input_camera_system_->getCameraShared(i).get(),
             pipelines[i]->getInputCameraShared().get());
    // Check that the output cameras actually point to the same object.
    CHECK_EQ(output_camera_system_->getCameraShared(i).get(),
             pipelines[i]->getOutputCameraShared().get());
  }
}

void SynchronousVisualNPipeline::processImage(
    const size_t camera_index, const cv::Mat& image,
    const int64_t timestamp_nanoseconds, VisualNFrame::Ptr* complete_nframe) {
  CHECK_NOTNULL(complete_nframe);

  CHECK_LE(camera_index, pipelines_.size());
  VisualFrame::Ptr frame =
      pipelines_[camera_index]->processImage(image, timestamp_nanoseconds);

  /// Create an iterator into the processing queue.
  std::map<int64_t, std::shared_ptr<VisualNFrame>>::iterator proc_it;
  bool create_new_nframe = false;
  if (processing_.empty()) {
    create_new_nframe = true;
  } else {
    // Try to find an existing NFrame in the processing list.
    // Use the timestamp of the frame because there may be a timestamp
    // corrector used in the pipeline.
    auto it_processing = processing_.lower_bound(
        frame->getTimestampNanoseconds());
    // Lower bound returns the first element that is not less than the value
    // (i.e. greater than or equal to the value).
    if (it_processing != processing_.begin()) { --it_processing; }
    // Now it_processing points to the first element that is less than the
    // value. Check both this value, and the one >=.
    int64_t min_time_diff_ns = std::abs(
        it_processing->first - frame->getTimestampNanoseconds());
    proc_it = it_processing;
    if (++it_processing != processing_.end()) {
      const int64_t time_diff_ns = std::abs(
          it_processing->first - frame->getTimestampNanoseconds());
      if (time_diff_ns < min_time_diff_ns) {
        proc_it = it_processing;
        min_time_diff_ns = time_diff_ns;
      }
    }
    // Now proc_it points to the closest nframes element.
    if (min_time_diff_ns > timestamp_tolerance_ns_) {
     create_new_nframe = true;
    }
  }

  if (create_new_nframe) {
    VisualNFrame::Ptr nframe(new VisualNFrame(output_camera_system_));
    bool not_replaced;
    std::tie(proc_it, not_replaced) = processing_.emplace(
        frame->getTimestampNanoseconds(), nframe);
    CHECK(not_replaced);
  }
  // Now proc_it points to the correct place in the processing_ list and
  // the NFrame has been created if necessary.
  VisualFrame::Ptr existing_frame = proc_it->second->getFrameShared(
      camera_index);
  if (existing_frame) {
    LOG(ERROR) << "Overwriting a frame at index " << camera_index
        << " with a new frame because the timestamp was the same.";
  }
  proc_it->second->setFrame(camera_index, frame);

  // Find the first index that has a complete nframe. Drop all incomplete
  // nframes before, because they won't ever be completed (because images arrive
  // in chronological order on every camera stream).
  TimestampVisualNFrameMap::reverse_iterator it_processing = processing_.rbegin();
  TimestampVisualNFrameMap::iterator delete_up_to_iterator = processing_.end();
  bool delete_nframes = false;
  while (it_processing != processing_.rend()) {
    CHECK(it_processing->second);
    if (it_processing->second->areAllFramesSet()) {
      *complete_nframe = it_processing->second;
      delete_up_to_iterator = it_processing.base();
      delete_nframes = true;
      break;
    }
    ++it_processing;
  }

  if (delete_nframes) {
    // Safety-check.
    bool complete_nframe_found = false;
    for (TimestampVisualNFrameMap::iterator it = processing_.begin();
        it != delete_up_to_iterator; ++it) {
      CHECK(it->second);
      CHECK(!complete_nframe_found);
      complete_nframe_found = it->second->areAllFramesSet();
    }
    CHECK(complete_nframe_found);
    processing_.erase(processing_.begin(), delete_up_to_iterator);
  }
}

std::shared_ptr<const NCamera> SynchronousVisualNPipeline::getInputNCameras() const {
  return input_camera_system_;
}

std::shared_ptr<const NCamera> SynchronousVisualNPipeline::getOutputNCameras() const {
  return output_camera_system_;
}

size_t SynchronousVisualNPipeline::getNumFramesProcessing() const {
  return processing_.size();
}

SynchronousVisualNPipeline::Ptr
SynchronousVisualNPipeline::createTestSynchronousVisualNPipeline(
    const size_t num_cameras, const int64_t timestamp_tolerance_ns) {
  NCamera::Ptr ncamera = NCamera::createTestNCamera(num_cameras);
  CHECK_EQ(ncamera->numCameras(), num_cameras);
  const bool kCopyImages = false;
  std::vector<VisualPipeline::Ptr> null_pipelines;
  for (size_t frame_idx = 0; frame_idx < num_cameras; ++frame_idx) {
    CHECK(ncamera->getCameraShared(frame_idx));
    null_pipelines.push_back(
        aligned_shared<NullVisualPipeline>(
            ncamera->getCameraShared(frame_idx), kCopyImages));
  }
  SynchronousVisualNPipeline::Ptr npipeline =
      aligned_shared<SynchronousVisualNPipeline>(
      null_pipelines, ncamera, ncamera, timestamp_tolerance_ns);
  return npipeline;
}
}  // namespace aslam
