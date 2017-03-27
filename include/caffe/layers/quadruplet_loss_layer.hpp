#ifndef CAFFE_QUADRUPLET_LOSS_LAYER_HPP_
#define CAFFE_QUADRUPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class QuadrupletLossLayer : public LossLayer<Dtype> {
 public:
  explicit QuadrupletLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_old_(), diff_new_() {}

  void Reshape(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top);

  inline const char* type() const { return "QuadrupletLoss"; }
  inline int ExactNumBottomBlobs() const { return 4; }
  inline bool AllowForceBackward(const int bottom_index) const { return true; }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

 protected:
  /// @copydoc QuadrupletLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_old_, diff_new_;
  Dtype alpha_, loss_old_, loss_new_;
  vector<Dtype> vec_loss_;
  int batch_size_;
  int vec_dimension_;
};

}  // namespace caffe

#endif  // CAFFE_QUADRUPLET_LOSS_LAYER_HPP_
