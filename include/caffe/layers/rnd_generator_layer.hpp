#ifndef CAFFE_RND_GENERATOR_LAYER_HPP_
#define CAFFE_RND_GENERATOR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class RndGeneratorLayer : public BaseDataLayer<Dtype> {
 public:
  explicit RndGeneratorLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RndGenerator"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  void InitRand();
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual int Rand(int n);

  int batch_size_, dim_, range_;
  Dtype scale_, shift_;
  shared_ptr<Caffe::RNG> rng_;
};

}  // namespace caffe

#endif  // CAFFE_MEMORY_DATA_LAYER_HPP_
