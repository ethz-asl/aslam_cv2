#include <aslam/pipeline/visual-npipeline.h>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/ncameras.h>
#include <aslam/common/thread-pool.h>
#include <aslam/frames/visual-nframes.h>
#include <aslam/pipeline/visual-pipeline.h>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>


namespace aslam {

VisualNPipeline::VisualNPipeline(
    unsigned num_threads,
    const std::vector<std::shared_ptr<VisualPipeline> >& pipelines,
    std::shared_ptr<NCameras> input_cameras,
    std::shared_ptr<NCameras> output_cameras, int64_t timestamp_tolerance_ns) :
      pipelines_(pipelines),
      input_cameras_(input_cameras), output_cameras_(output_cameras),
      timestamp_tolerance_ns_(timestamp_tolerance_ns) {
  // Defensive programming ninjitsu.
  CHECK_NOTNULL(input_cameras.get());
  CHECK_NOTNULL(output_cameras.get());
  CHECK_GT(input_cameras->numCameras(), 0u);
  CHECK_EQ(input_cameras->numCameras(), output_cameras->numCameras());
  CHECK_EQ(input_cameras->numCameras(), pipelines.size());
  CHECK_GE(timestamp_tolerance_ns, 0);

  for(size_t i = 0; i < pipelines.size(); ++i) {
    CHECK_NOTNULL(pipelines[i].get());
    // Check that the input cameras actually point to the same object.
    CHECK_EQ(input_cameras->getCameraMutable(i), pipelines[i]->getInputCamera());
    // Check that the output cameras actually point to the same object.
    CHECK_EQ(output_cameras->getCameraMutable(i), pipelines[i]->getOutputCamera());
  }
  CHECK_GT(num_threads, 0u);
  thread_pool_.reset(new ThreadPool(num_threads));
}

VisualNPipeline::~VisualNPipeline() {
  thread_pool_->stop();
}

void VisualNPipeline::processImage(size_t camera_index, const cv::Mat& image,
                                   int64_t system_stamp, int64_t hardware_stamp) {
  thread_pool_->enqueue(&VisualNPipeline::work, this, camera_index,
                        image, system_stamp, hardware_stamp);
}

size_t VisualNPipeline::getNumFramesComplete() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return completed_.size();
}

std::shared_ptr<VisualNFrames> VisualNPipeline::getNext() {
  // Initialize the return value as null
  std::shared_ptr<VisualNFrames> rval;
  std::unique_lock<std::mutex> lock(mutex_);
  if(!completed_.empty()) {
    /// Get the oldest frame.
    auto it = completed_.begin();
    rval = it->second;
    completed_.erase(it);
  }
  return rval;
}

std::shared_ptr<VisualNFrames> VisualNPipeline::getLatestAndClear() {
  std::shared_ptr<VisualNFrames> rval;
  std::unique_lock<std::mutex> lock(mutex_);
  if(!completed_.empty()) {
    /// Get the latest frame.
    auto it = completed_.rbegin();
    rval = it->second;
    int64_t timestamp = it->first;
    completed_.clear();
    // Clear any processing frames older than this one.
    auto pit = processing_.begin();
    while(pit != processing_.end() && pit->first <= timestamp) {
      pit = processing_.erase(pit);
    }
  }
  return rval;
}

const std::shared_ptr<NCameras>& VisualNPipeline::getInputNCameras() const {
  return input_cameras_;
}

const std::shared_ptr<NCameras>& VisualNPipeline::getOutputNCameras() const {
  return output_cameras_;
}

size_t VisualNPipeline::getNumFramesProcessing() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return processing_.size();
}

void VisualNPipeline::work(size_t camera_index, const cv::Mat& image,
                           int64_t system_stamp, int64_t hardware_stamp) {
  std::shared_ptr<VisualFrame> frame;
  frame = pipelines_[camera_index]->processImage(image, system_stamp, hardware_stamp);

  /// Create an iterator into the processing queue.
  std::map<int64_t, std::shared_ptr<VisualNFrames>>::iterator proc_it;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    bool create_new_nframes = false;
    if(processing_.empty()) {
      create_new_nframes = true;
    } else {
      // Try to find an existing NFrame in the processing list.
      // Use the timestamp of the frame because there may be a timestamp
      // corrector used in the pipeline.
      auto it = processing_.lower_bound(frame->getTimestamp());
      // Lower bound returns the first element that is not less than the value
      // (i.e. greater than or equal to the value).
      if(it != processing_.begin()) { --it; }
      // Now it points to the first element that is less than the value.
      // Check both this value, and the one >=.
      int64_t min_time_diff = std::abs(it->first - frame->getTimestamp());
      proc_it = it;
      if(++it != processing_.end()) {
        int64_t time_diff = std::abs(it->first - frame->getTimestamp());
        if(time_diff < min_time_diff) {
          proc_it = it;
          min_time_diff = time_diff;
        }
      }
      // Now proc_it points to the closest nframes element.
      if(min_time_diff > timestamp_tolerance_ns_) {
       create_new_nframes = true;
      }
    }

    if(create_new_nframes) {
      std::shared_ptr<VisualNFrames> nframes(new VisualNFrames(output_cameras_));
      bool replaced;
      std::tie(proc_it, replaced) = processing_.insert(
          std::make_pair(frame->getTimestamp(), nframes)
      );
    }
    // Now proc_it points to the correct place in the processing_ list and
    // the NFrame has been created if necessary.
    VisualFrame::Ptr existing_frame = proc_it->second->getFrameMutable(camera_index);
    if(existing_frame) {
      LOG(WARNING) << "Overwriting a frame at index " << camera_index << ":\n"
          << *existing_frame << "\nwith a new frame: "
          << *frame << "\nbecause the timestamp was the same.";
    }
    proc_it->second->setFrame(camera_index, frame);

    // Check if all images have been received.
    bool all_received = true;
    for(size_t i = 0; i < proc_it->second->getNumFrames(); ++i) {
      all_received &= proc_it->second->isFrameNull(i);
    }

    if(all_received) {
      completed_.insert(*proc_it);
      processing_.erase(proc_it);
    }
  }
}

void VisualNPipeline::waitForAllWorkToComplete() const {
  thread_pool_->waitForEmptyQueue();
}
}  // namespace aslam
