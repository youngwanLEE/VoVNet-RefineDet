// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//ywlee
#include "caffe/layers/detection_output_layer.hpp"
#include <iostream>
#include <ctime>
#include <cstdio>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value,
           const int& device_id);

  // std::vector<vector<float> > Detect(const cv::Mat& img);
  std::vector<vector<float> > Detect(const cv::Mat& img, double &pTime);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value,
                   const int& device_id) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(device_id);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  // input_geometry_ = cv::Size(1024, 512);

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img, double &pTime) {

  //ywlee
  double startTime=.0;

  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);


  startTime = std::clock();

  net_->Forward();

  pTime = (std::clock() - startTime)/(double)CLOCKS_PER_SEC;

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "video",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.5,
    "Only store detections with score higher than the threshold.");
//ywlee
DEFINE_string(label_map_file, "data/coco/labelmap_coco.prototxt",
    "labelmap_file");



int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& file_type = FLAGS_file_type;
  const string& out_file = FLAGS_out_file;
  const float confidence_threshold = FLAGS_confidence_threshold;
  //ywlee
  static cv::VideoWriter cap_out;
  const string& label_map_file = FLAGS_label_map_file;
  const bool& save_true = true;
  const string& save_file = argv[4]; // 1088x1920(544x960)
  // const string& save_file = "data/RefineDet_hyper_v3_D2_g32_S0-6-S0021.avi"; // 1088x1920(544x960)
  int device_id = 0;
  if (argc == 6){
    device_id = atoi(argv[5]);
  }



  LabelMap label_map;
  map<int, string> label_to_name_;
  map<int, string> label_to_display_name_;

  CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
      << "Failed to read label map file: " << label_map_file;
  CHECK(MapLabelToName(label_map, true, &label_to_name_))
      << "Failed to convert label to name.";
  CHECK(MapLabelToDisplayName(label_map, true, &label_to_display_name_))
      << "Failed to convert label to display name.";

  vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());


  //timer
  std::clock_t start;
  double duration=0.;
  double dTimeSum=0.;
  double dAvgTime=0.;
  int nCnt=0;
  char buffer[50];
  string label_name = "Unknown";
  int fontface = cv::FONT_HERSHEY_SIMPLEX;
  double scale =0.5;
  int thickness = 0.8;
  int baseline = 0;

  int baseline_fps = 2;
  double scale_fps = 2;
  int thickness_fps = 1;


  // Initialize the network.
  // Detector detector(model_file, weights_file, mean_file, mean_value);
  Detector detector(model_file, weights_file, mean_file, mean_value, device_id);

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ofstream outfile;
  if (!out_file.empty()) {
    outfile.open(out_file.c_str());
    if (outfile.good()) {
      buf = outfile.rdbuf();
    }
  }
  std::ostream out(buf);

  // Process image one by one.
  std::ifstream infile(argv[3]);
  std::string file;
  while (infile >> file) {

    nCnt++;
    start = std::clock();

    if (file_type == "image") {
      cv::Mat img = cv::imread(file, -1);
      CHECK(!img.empty()) << "Unable to decode image " << file;
      std::vector<vector<float> > detections = detector.Detect(img, duration);

      /* Print the detection results. */
      for (int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        if (score >= confidence_threshold) {
          out << file << " ";
          out << static_cast<int>(d[1]) << " ";
          out << score << " ";
          out << static_cast<int>(d[3] * img.cols) << " ";
          out << static_cast<int>(d[4] * img.rows) << " ";
          out << static_cast<int>(d[5] * img.cols) << " ";
          out << static_cast<int>(d[6] * img.rows) << std::endl;
        }
      }
    } else if (file_type == "video") {
      cv::VideoCapture cap(file);
      if (!cap.isOpened()) {
        LOG(FATAL) << "Failed to open video: " << file;
      }
      cv::Mat img;
      int frame_count = 1;
      while (true) {
        bool success = cap.read(img);
        if (!success) {
          LOG(INFO) << "Process " << frame_count << " frames from " << file;
          break;
        }
        CHECK(!img.empty()) << "Error when read frame";

        nCnt++;
        // start = std::clock();

        std::vector<vector<float> > detections = detector.Detect(img, duration);

        //ywlee
        // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        dTimeSum += duration;
        out<<"frame : "<<nCnt<<"  pTime : "<<duration<<std::endl;


        /* Print the detection results. */
        for (int i = 0; i < detections.size(); ++i) {
          const vector<float>& d = detections[i];
          // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
          CHECK_EQ(d.size(), 7);
          const float score = d[2];
          if(score >= confidence_threshold)
          {
            int label = static_cast<int>(d[1]);
            label_name = label_to_display_name_.find(label)->second;
            CHECK_LT(label, colors.size());
            const cv::Scalar& color = colors[label];
            cv::Point top_left_pt(static_cast<int>(d[3] * img.cols),static_cast<int>(d[4] * img.rows));
            cv::Point bottom_right_pt(static_cast<int>(d[5] * img.cols),static_cast<int>(d[6] * img.rows));
            cv::rectangle(img, top_left_pt, bottom_right_pt,color,2.5);

            cv::Point bottom_left_pt(static_cast<int>(d[3] * img.cols),static_cast<int>(d[6] * img.rows));
            snprintf(buffer, sizeof(buffer), "%s: %.2f", label_name.c_str(), score);
            cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness, &baseline);

            cv::rectangle(img, bottom_left_pt+cv::Point(0,0),
                          bottom_left_pt+cv::Point(text.width, -text.height-baseline),
                          color, CV_FILLED);
            cv::putText(img, buffer, bottom_left_pt - cv::Point(0, baseline), fontface, scale, CV_RGB(0,0,0), thickness, 8);

            //show FPS
            // snprintf(buffer, sizeof(buffer), "FPS:%.2f", 1/duration);
            // cv::Size text_fps = cv::getTextSize(buffer, fontface, scale, thickness, &baseline);
            // cv::rectangle(img, cv::Point(0, 0),
            //                 cv::Point(text_fps.width, text_fps.height + baseline),
            //                 CV_RGB(255, 255, 255), CV_FILLED);
            // cv::putText(img, buffer, cv::Point(0, text_fps.height + baseline / 2.),
            // fontface, scale, CV_RGB(0, 0, 0), thickness, 8);

            //show FPS
            // snprintf(buffer, sizeof(buffer), "FPS:%.2f", 1/duration);
            // cv::Size text_fps = cv::getTextSize(buffer, fontface, scale_fps, thickness_fps, &baseline_fps);
            // cv::rectangle(img, cv::Point(0, 0),
            //                 cv::Point(text_fps.width, text_fps.height + baseline_fps),
            //                 CV_RGB(255, 255, 255), CV_FILLED);
            // cv::putText(img, buffer, cv::Point(0, text_fps.height + baseline_fps / 2.),
            // fontface, scale_fps, CV_RGB(0, 0, 0), thickness_fps, 8);

          }

        }
        // Save result if required.
        if(save_true)
        {
          if (!cap_out.isOpened()) {
            cv::Size size(img.size().width, img.size().height);
            printf("width : %d, Height : %d\n", size.width, size.height);
            cv::VideoWriter outputVideo(save_file,  CV_FOURCC('M', 'J', 'P', 'G'),
                                        20, size, true);
            cap_out = outputVideo;
          }
          // if(frame_count % 3 == 0){
          //   cap_out.write(img);
          // }
          cap_out.write(img);

        }

        ++frame_count;
        // cv::imshow("Detection results",img);
        // cv::waitKey(1);
      }


      if (cap.isOpened()) {
        cap.release();
      }
    } else {
      LOG(FATAL) << "Unknown file_type: " << file_type;
    }
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
