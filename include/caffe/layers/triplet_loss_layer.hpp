#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class TripletLossLayer : public LossLayer<Dtype> {
 public:
  explicit TripletLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_same_class_(), diff_diff_class_(), semi_hard_(false), start_idx(0) {}

  void Reshape(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top);

  inline const char* type() const { return "TripletLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 4; }
  inline bool AllowForceBackward(const int bottom_index) const { return true; }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

 protected:
  /// @copydoc TripletLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_same_class_;
  Blob<Dtype> diff_diff_class_;
  Dtype mining_ratio_;
  Dtype alpha_;
  bool intriplet_mining_;
  bool semi_hard_;
  vector<Dtype> vec_loss_;
  int batch_size_;
  int vec_dimension_;
  int start_idx;
};

}  // namespace caffe

#endif  // CAFFE_TRIPLET_LOSS_LAYER_HPP_
