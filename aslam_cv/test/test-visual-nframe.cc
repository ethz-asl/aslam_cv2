#include <eigen-checks/gtest.h>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <aslam/common/memory.h>
#include <aslam/common/unique-id.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <aslam/cameras/camera-pinhole.h>

TEST(NFrame, MinTimestamp) {
  aslam::NCamera::Ptr ncamera = aslam::NCamera::createSurroundViewTestNCamera();
  aslam::VisualFrame::Ptr frame_0(new aslam::VisualFrame);
  frame_0->setCameraGeometry(ncamera->getCameraShared(0));
  frame_0->setTimestampNanoseconds(123);
  aslam::VisualFrame::Ptr frame_1(new aslam::VisualFrame);
  frame_1->setCameraGeometry(ncamera->getCameraShared(1));
  frame_1->setTimestampNanoseconds(341566);
  aslam::VisualFrame::Ptr frame_2(new aslam::VisualFrame);
  frame_2->setCameraGeometry(ncamera->getCameraShared(2));
  frame_2->setTimestampNanoseconds(98);
  aslam::VisualFrame::Ptr frame_3(new aslam::VisualFrame);
  frame_3->setCameraGeometry(ncamera->getCameraShared(3));
  frame_3->setTimestampNanoseconds(5);
  aslam::NFramesId nframe_id;
  nframe_id.randomize();
  aslam::VisualNFrame nframe(nframe_id, 4);
  nframe.setNCameras(ncamera);
  nframe.setFrame(0, frame_0);
  nframe.setFrame(1, frame_1);
  nframe.setFrame(2, frame_2);
  nframe.setFrame(3, frame_3);
  int64_t min_timestamp = nframe.getMinTimestampNanoseconds();
  ASSERT_EQ(min_timestamp, 5);
}

ASLAM_UNITTEST_ENTRYPOINT
