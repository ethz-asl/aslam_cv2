#include <aslam/pipeline/visual-npipeline.h>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/common/memory.h>
#include <aslam/common/thread-pool.h>
#include <aslam/frames/visual-nframe.h>
#include <aslam/pipeline/visual-pipeline.h>
#include <aslam/pipeline/visual-pipeline-null.h>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>

namespace aslam {

VisualNPipeline::VisualNPipeline(
    size_t num_threads,
    const std::vector<std::shared_ptr<VisualPipeline> >& pipelines,
    const std::shared_ptr<NCamera>& input_camera_system,
    const std::shared_ptr<NCamera>& output_camera_system, int64_t timestamp_tolerance_ns) :
      pipelines_(pipelines),
      shutdown_(false),
      input_camera_system_(input_camera_system), output_camera_system_(output_camera_system),
      timestamp_tolerance_ns_(timestamp_tolerance_ns)  {
  // Defensive programming ninjitsu.
  CHECK_NOTNULL(input_camera_system_.get());
  CHECK_NOTNULL(output_camera_system.get());
  CHECK_GT(input_camera_system_->numCameras(), 0u);
  CHECK_EQ(input_camera_system_->numCameras(), output_camera_system_->numCameras());
  CHECK_EQ(input_camera_system_->numCameras(), pipelines.size());
  CHECK_GE(timestamp_tolerance_ns, 0);

  for (size_t i = 0; i < pipelines.size(); ++i) {
    CHECK_NOTNULL(pipelines[i].get());
    // Check that the input cameras actually point to the same object.
    CHECK_EQ(input_camera_system_->getCameraShared(i).get(),
             pipelines[i]->getInputCameraShared().get());
    // Check that the output cameras actually point to the same object.
    CHECK_EQ(output_camera_system_->getCameraShared(i).get(),
             pipelines[i]->getOutputCameraShared().get());
  }
  CHECK_GT(num_threads, 0u);
  thread_pool_.reset(new ThreadPool(num_threads));
}

VisualNPipeline::~VisualNPipeline() {
  shutdown();
}

void VisualNPipeline::shutdown() {
  shutdown_ = true;
  condition_not_empty_.notify_all();
  condition_not_full_.notify_all();
  thread_pool_->stop();
}

bool VisualNPipeline::processImageBlockingIfFull(size_t camera_index, const cv::Mat& image,
                                                 int64_t timestamp, size_t max_queue_size) {
  while (!shutdown_) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed_.size() >= max_queue_size) {
      condition_not_full_.wait(lock);
      if (completed_.size() >= max_queue_size) {
        continue;
      }
    }
    processImage(camera_index, image, timestamp);
    return true;
  }
  return false;
}

bool VisualNPipeline::getNextBlocking(std::shared_ptr<VisualNFrame>* nframe) {
  CHECK_NOTNULL(nframe);
  while (!shutdown_) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed_.empty()) {
      condition_not_empty_.wait(lock);
    }
    if (completed_.empty()) {
      continue;
    }

    // Get the oldest frame.
    lock.unlock();
    *nframe = getNext();
    CHECK(*nframe);
    return true;
  }
  // Queue shutdown.
  return false;
}

void VisualNPipeline::processImage(size_t camera_index, const cv::Mat& image, int64_t timestamp) {
  thread_pool_->enqueue(&VisualNPipeline::work, this, camera_index, image, timestamp);
}

size_t VisualNPipeline::getNumFramesComplete() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return completed_.size();
}

std::shared_ptr<VisualNFrame> VisualNPipeline::getNext() {
  // Initialize the return value as null
  std::shared_ptr<VisualNFrame> rval;
  std::unique_lock<std::mutex> lock(mutex_);
  if(!completed_.empty()) {
    // Get the oldest frame.
    auto it = completed_.begin();
    rval = it->second;
    completed_.erase(it);
    condition_not_full_.notify_all();
  }
  return rval;
}

std::shared_ptr<VisualNFrame> VisualNPipeline::getLatestAndClear() {
  std::shared_ptr<VisualNFrame> rval;
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

std::shared_ptr<const NCamera> VisualNPipeline::getInputNCameras() const {
  return input_camera_system_;
}

std::shared_ptr<const NCamera> VisualNPipeline::getOutputNCameras() const {
  return output_camera_system_;
}

size_t VisualNPipeline::getNumFramesProcessing() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return processing_.size();
}

void VisualNPipeline::work(size_t camera_index, const cv::Mat& image, int64_t timestamp) {
  CHECK_LE(camera_index, pipelines_.size());
  std::shared_ptr<VisualFrame> frame;
  frame = pipelines_[camera_index]->processImage(image, timestamp);

  /// Create an iterator into the processing queue.
  std::map<int64_t, std::shared_ptr<VisualNFrame>>::iterator proc_it;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    bool create_new_nframes = false;
    if(processing_.empty()) {
      create_new_nframes = true;
    } else {
      // Try to find an existing NFrame in the processing list.
      // Use the timestamp of the frame because there may be a timestamp
      // corrector used in the pipeline.
      auto it = processing_.lower_bound(frame->getTimestampNanoseconds());
      // Lower bound returns the first element that is not less than the value
      // (i.e. greater than or equal to the value).
      if(it != processing_.begin()) { --it; }
      // Now it points to the first element that is less than the value.
      // Check both this value, and the one >=.
      int64_t min_time_diff = std::abs(it->first - frame->getTimestampNanoseconds());
      proc_it = it;
      if(++it != processing_.end()) {
        int64_t time_diff = std::abs(it->first - frame->getTimestampNanoseconds());
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
      std::shared_ptr<VisualNFrame> nframes(new VisualNFrame(output_camera_system_));
      bool replaced;
      std::tie(proc_it, replaced) = processing_.insert(
          std::make_pair(frame->getTimestampNanoseconds(), nframes)
      );
    }
    // Now proc_it points to the correct place in the processing_ list and
    // the NFrame has been created if necessary.
    VisualFrame::Ptr existing_frame = proc_it->second->getFrameShared(camera_index);
    if(existing_frame) {
      LOG(WARNING) << "Overwriting a frame at index " << camera_index << ":\n"
          << *existing_frame << "\nwith a new frame: "
          << *frame << "\nbecause the timestamp was the same.";
    }
    proc_it->second->setFrame(camera_index, frame);

    // Move all completed nframes from the processed_ queue to the completed_ queue chronologically.
    auto it_processing = processing_.begin();
    while (it_processing != processing_.end()) {
      // Check if all images have been received.
      bool all_received = true;
      for (size_t i = 0; i < it_processing->second->getNumFrames(); ++i) {
        all_received &= it_processing->second->isFrameSet(i);
      }

      if (all_received) {
        completed_.insert(*it_processing);
        it_processing = processing_.erase(it_processing);
        condition_not_empty_.notify_all();
      } else {
        // As we are iterating over the map in chronological order we have to abort once an nframe
        // is not yet finished processing to keep chornological ordering in the destination queue.
        break;
      }
    }
  }
}

void VisualNPipeline::waitForAllWorkToComplete() const {
  thread_pool_->waitForEmptyQueue();
}

VisualNPipeline::Ptr VisualNPipeline::createTestVisualNPipeline(size_t num_cameras,
                                                                size_t num_threads,
                                                                int64_t timestamp_tolerance_ns) {
  NCamera::Ptr ncamera = NCamera::createTestNCamera(num_cameras);
  CHECK_EQ(ncamera->numCameras(), num_cameras);
  const bool kCopyImages = false;
  std::vector<VisualPipeline::Ptr> null_pipelines;
  for (size_t frame_idx = 0; frame_idx < num_cameras; ++frame_idx) {
    CHECK(ncamera->getCameraShared(frame_idx));
    null_pipelines.push_back(
       aslam::aligned_shared<NullVisualPipeline>(ncamera->getCameraShared(frame_idx), kCopyImages));
  }
  VisualNPipeline::Ptr npipeline = aslam::aligned_shared<VisualNPipeline>(num_threads,
                                                                          null_pipelines,
                                                                          ncamera,
                                                                          ncamera,
                                                                          timestamp_tolerance_ns);
  return npipeline;
}
}  // namespace aslam
