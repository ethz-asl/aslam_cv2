#include <visensor/visensor.hpp>


void imageCallback(visensor::ViFrame::Ptr frame_ptr) {
  uint32_t camera_id = frame_ptr->camera_id;
  //push image on queue
  frameQueue[camera_id].push(frame_ptr);
}


int main()
{


  // Connect to sensor.
  try {
    visensor.init();
  } catch (visensor::exceptions::ConnectionException const &ex) {
    std::cout << ex.what() << "\n";
    return 0;
  }

  // set callback for image messages
  visensor.setCameraCallback(boost::bind(&ViSensorInterface::ImageCallback, this, _1));
  visensor.startAllCameras(image_rate);


}



//
//void ViSensorInterface::worker(unsigned int cam_id) {
//  while (1) {
//    //Popping image from queue. If no image available, we perform a blocking wait
//    visensor::ViFrame::Ptr frame = frameQueue[cam_id].pop();
//    uint32_t camera_id = frame->camera_id;
//    cv::Mat image;
//    image.create(frame->height, frame->width, CV_8UC1);
//    memcpy(image.data, frame->getImageRawPtr(), frame->height * frame->width);
//    //update window with image
//    char winName[255];
//    boost::mutex::scoped_lock lock(io_mutex_);  //lock thread as opencv does seem to have problems with multithreading
//    sprintf(winName, "Camera %u", camera_id);
//    cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
//    cv::imshow(winName, image);
//    cv::waitKey(1);
//  }
//}
