#ifndef CAFFE_DATA_OUTPUT_LAYER_HPP_
#define CAFFE_DATA_OUTPUT_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DataOutputLayer : public Layer<Dtype> {
 public:
  explicit DataOutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {};
  virtual ~DataOutputLayer() {};
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DataOutput"; }
  virtual inline int MinNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  { NOT_IMPLEMENTED; };

  std::vector<std::string> file_name;
  std::vector<int> bottom_num;
  std::vector<int> bottom_num_stride;
  DataOutputParameter_FileType output_type;
};

}  // namespace caffe

#endif  // CAFFE_DATA_OUTPUT_LAYER_HPP_
