#include <gtest/gtest.h>
#include <aslam/cameras/camera.h>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/cameras/ncameras.h>
#include <aslam/frames/visual-nframes.h>
#include <aslam/pipeline/visual-pipeline-null.h>
#include <aslam/pipeline/visual-pipeline.h>
#include <aslam/pipeline/visual-npipeline.h>
#include <opencv2/core/core.hpp>

using namespace aslam;

class VisualNPipelineTest : public ::testing::Test {
 protected:
  typedef aslam::RadTanDistortion DistortionType;
  typedef aslam::PinholeCamera CameraType;

  virtual void SetUp() { }

  void constructNCameras(unsigned num_cameras,
                         unsigned num_threads,
                         int64_t timestamp_tolerance_ns) {
    //random intrinsics
    double fu = 300;
    double fv = 320;
    double cu = 340;
    double cv = 220;
    double res_u = 640;
    double res_v = 480;

    Eigen::VectorXd intrinsics(4);
    intrinsics << fu, fv, cu, cv;

    //random radtan distortion parameters
    Eigen::VectorXd distortion_param(4);
    distortion_param << 0.8, 0.01, 0.2, 0.02;

    NCamerasId id;
    id.randomize();
    Aligned<std::vector, kindr::minimal::QuatTransformation>::type T_C_B;
    std::vector<Camera::Ptr> cameras;
    std::vector<std::shared_ptr<VisualPipeline>> pipelines;
    for(unsigned i = 0; i < num_cameras; ++i) {
      kindr::minimal::QuatTransformation T;
      T_C_B.push_back(T);

      std::shared_ptr<Camera> camera = Camera::construct<CameraType, DistortionType>(
          intrinsics, res_u, res_v, distortion_param);
      cameras.push_back(camera);
      pipelines.push_back( std::shared_ptr<VisualPipeline>(new NullVisualPipeline(camera, false)));
    }
    camera_rig_.reset( new NCameras(id, T_C_B, cameras, "Test Camera System"));

    pipeline_.reset( new VisualNPipeline(num_threads, pipelines,
                                         camera_rig_, camera_rig_,
                                         timestamp_tolerance_ns));
  }

  cv::Mat getImageFromCamera(unsigned camera_index) {
    CHECK_NOTNULL(camera_rig_.get());
    CHECK_LT(camera_index, camera_rig_->getNumCameras());
    return cv::Mat(this->camera_rig_->getCamera(camera_index).imageHeight(),
                   this->camera_rig_->getCamera(camera_index).imageWidth(),
                   CV_8UC1, uint8_t(camera_index));
  }

  std::shared_ptr<NCameras> camera_rig_;
  std::shared_ptr<VisualNPipeline> pipeline_;

};

TEST_F(VisualNPipelineTest, buildNFramesOutOfOrder) {
  this->constructNCameras(2, 4, 100);

  // Build n frames out of order.
  pipeline_->processImage(0, getImageFromCamera(0), 0, 0);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(0u, pipeline_->getNumFramesComplete());

  pipeline_->processImage(0, getImageFromCamera(0), 1000, 1000);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(0u, pipeline_->getNumFramesComplete());

  pipeline_->processImage(1, getImageFromCamera(0), 1, 1);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(1u, pipeline_->getNumFramesComplete());

  pipeline_->processImage(1, getImageFromCamera(0), 1001, 1001);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(2u, pipeline_->getNumFramesComplete());

  std::shared_ptr<VisualNFrames> nframes = pipeline_->getNext();

  ASSERT_EQ(1u, pipeline_->getNumFramesComplete());
  ASSERT_TRUE(nframes.get() != NULL);
  ASSERT_EQ(0, nframes->getFrame(0).getTimestamp());
  ASSERT_EQ(1, nframes->getFrame(1).getTimestamp());

  nframes = pipeline_->getNext();
  ASSERT_EQ(0u, pipeline_->getNumFramesComplete());
  ASSERT_TRUE(nframes.get() != NULL);
  ASSERT_EQ(1000, nframes->getFrame(0).getTimestamp());
  ASSERT_EQ(1001, nframes->getFrame(1).getTimestamp());
}


TEST_F(VisualNPipelineTest, testBuildAndClear) {
  this->constructNCameras(2, 4, 100);

  // Check that the clearing of older completed frames works.
  pipeline_->processImage(0, getImageFromCamera(0), 0, 0);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(0u, pipeline_->getNumFramesComplete());

  pipeline_->processImage(0, getImageFromCamera(0), 1000, 1000);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(0u, pipeline_->getNumFramesComplete());

  pipeline_->processImage(1, getImageFromCamera(0), 1, 1);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(1u, pipeline_->getNumFramesComplete());

  pipeline_->processImage(1, getImageFromCamera(0), 1001, 1001);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(2u, pipeline_->getNumFramesComplete());

  std::shared_ptr<VisualNFrames> nframes = pipeline_->getLatestAndClear();

  ASSERT_EQ(0u, pipeline_->getNumFramesComplete());
  ASSERT_TRUE(nframes.get() != NULL);
  ASSERT_EQ(1000, nframes->getFrame(0).getTimestamp());
  ASSERT_EQ(1001, nframes->getFrame(1).getTimestamp());
}

TEST_F(VisualNPipelineTest, testTimestampDiff) {
  this->constructNCameras(2, 4, 100);

  // Check that the timestamp tolerance is respected.
  pipeline_->processImage(0, getImageFromCamera(0), 0, 0);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(0u, pipeline_->getNumFramesComplete());

  pipeline_->processImage(1, getImageFromCamera(1), 100, 100);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(1u, pipeline_->getNumFramesComplete());
  std::shared_ptr<VisualNFrames> nframes = pipeline_->getLatestAndClear();
  ASSERT_TRUE(nframes.get() != NULL);
  ASSERT_EQ(0, nframes->getFrame(0).getTimestamp());
  ASSERT_EQ(100, nframes->getFrame(1).getTimestamp());

  // Build n frames out of order.
  pipeline_->processImage(0, getImageFromCamera(0), 0, 0);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(0u, pipeline_->getNumFramesComplete());

  pipeline_->processImage(1, getImageFromCamera(1), 101, 101);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(0u, pipeline_->getNumFramesComplete());
  ASSERT_EQ(2u, pipeline_->getNumFramesProcessing());

  // Get the latest. This should be null as no frames are complete.
  nframes = pipeline_->getLatestAndClear();
  ASSERT_TRUE(nframes.get() == NULL);
  // And there should still be two processing
  ASSERT_EQ(2u, pipeline_->getNumFramesProcessing());

  // Add an even later frame
  pipeline_->processImage(0, getImageFromCamera(0), 401, 401);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(3u, pipeline_->getNumFramesProcessing());
  ASSERT_EQ(0u, pipeline_->getNumFramesComplete());

  // Finish the middle frame
  pipeline_->processImage(0, getImageFromCamera(0), 101, 101);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(2u, pipeline_->getNumFramesProcessing());
  ASSERT_EQ(1u, pipeline_->getNumFramesComplete());
  // This should clear the oldest unfinished frame from the processing queue
  nframes = pipeline_->getLatestAndClear();
  ASSERT_TRUE(nframes.get() != NULL);
  ASSERT_EQ(101, nframes->getFrame(0).getTimestamp());
  ASSERT_EQ(101, nframes->getFrame(1).getTimestamp());
  ASSERT_EQ(1u, pipeline_->getNumFramesProcessing());
  ASSERT_EQ(0u, pipeline_->getNumFramesComplete());

  // Finish the last processing frame
  pipeline_->processImage(1, getImageFromCamera(1), 401, 401);
  pipeline_->waitForAllWorkToComplete();
  ASSERT_EQ(0u, pipeline_->getNumFramesProcessing());
  ASSERT_EQ(1u, pipeline_->getNumFramesComplete());
  nframes = pipeline_->getLatestAndClear();
  ASSERT_TRUE(nframes.get() != NULL);
  ASSERT_EQ(401, nframes->getFrame(0).getTimestamp());
  ASSERT_EQ(401, nframes->getFrame(1).getTimestamp());
}
