#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <omp.h>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/flexible_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <boost/algorithm/string.hpp>

namespace caffe {

template <typename Dtype>
FlexibleImageDataLayer<Dtype>::~FlexibleImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void FlexibleImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.flexible_image_data_param().new_height();
  const int new_width  = this->layer_param_.flexible_image_data_param().new_width();
  const bool is_color  = this->layer_param_.flexible_image_data_param().is_color();
  const int img_num = this->layer_param_.flexible_image_data_param().img_num();
  const int label_num = this->layer_param_.flexible_image_data_param().label_num();
  string root_folder = this->layer_param_.flexible_image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.flexible_image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  vector<string> image_path(img_num);
  vector<float> label(label_num);
  // multi-img and multi-label support
  while (std::getline(infile, line)) {
    vector<string> strs;
    line.erase(line.find_last_not_of("\t ") + 1);
    boost::split(strs, line, boost::is_any_of("\t "));
    CHECK_EQ(img_num + label_num, strs.size()) <<
        "Plz double check your sample list has the exact image and label number as specified";
    for (int i = 0; i < img_num; i++) {
        image_path[i] = strs[i];
    }
    for (int i = 0; i < label_num; i++) {
        label[i] = atof(strs[i + img_num].c_str());
    }
    lines_.push_back(std::make_pair(image_path, label));
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.flexible_image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  } else {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
        this->layer_param_.flexible_image_data_param().rand_skip() == 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.flexible_image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.flexible_image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first[0],
          new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first[0];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);

  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.flexible_image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  top_shape[1] *= img_num;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  LOG(INFO) << "input data label size: " << label_num;
  // label
  vector<int> label_shape(4, 1);
  label_shape[0] = batch_size;
  label_shape[1] = label_num;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void FlexibleImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void FlexibleImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  FlexibleImageDataParameter flexible_image_data_param = this->layer_param_.flexible_image_data_param();
  const int batch_size = flexible_image_data_param.batch_size();
  const int new_height = flexible_image_data_param.new_height();
  const int new_width = flexible_image_data_param.new_width();
  const bool is_color = flexible_image_data_param.is_color();
  const int img_num = flexible_image_data_param.img_num();
  const int label_num = flexible_image_data_param.label_num();
  const int thread_num = flexible_image_data_param.thread_num();
  string root_folder = flexible_image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first[0],
          new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first[0];
  CHECK_LE(thread_num, img_num) << "No need to use redundant threads which cannot improve the efficiency.";
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  top_shape[1] *= img_num;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

#pragma omp parallel num_threads(thread_num)
    {
#pragma omp for
    for (int img_id = 0; img_id < img_num; img_id++) {
        cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first[img_id],
                new_height, new_width, is_color);
        CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first[img_id];
        read_time += timer.MicroSeconds();
        timer.Start();

        // Apply transformations (mirror, crop...) to the image
        int offset = batch->data_.offset(item_id, cv_img.channels() * img_id);
        this->transformed_data_.set_cpu_data(prefetch_data + offset);
        this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
        trans_time += timer.MicroSeconds();
        if (img_id != img_num - 1)
            timer.Start();
    }
    }
    for (int label_id = 0; label_id < label_num; label_id++) {
        prefetch_label[item_id*label_num + label_id] = lines_[lines_id_].second[label_id];
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.flexible_image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(FlexibleImageDataLayer);
REGISTER_LAYER_CLASS(FlexibleImageData);

}  // namespace caffe
#endif  // USE_OPENCV
