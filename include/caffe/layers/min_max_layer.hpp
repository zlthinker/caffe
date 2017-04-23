#ifndef CAFFE_MIN_MAX_LAYER_HPP_
#define CAFFE_MIN_MAX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Rescale the input linearly to 0~1.
 *
 */
template <typename Dtype>
class MinMaxLayer : public Layer<Dtype> {
 public:
  explicit MinMaxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MVN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> min_, max_;
  Blob<Dtype> gap_;
};

}  // namespace caffe

#endif  // CAFFE_MIN_MAX_LAYER_HPP_
