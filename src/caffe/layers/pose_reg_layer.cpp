#include <vector>

#include "caffe/layers/pose_reg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PoseRegLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  beta_ = this->layer_param_.threshold_param().threshold();
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Trans inputs must have the same dimension.";
  CHECK_EQ(bottom[2]->count(1), bottom[3]->count(1))
      << "Rot inputs must have the same dimension.";
  rot_diff_.ReshapeLike(*bottom[0]);
  trans_diff_.ReshapeLike(*bottom[2]);
}

template <typename Dtype>
void PoseRegLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // calc trans loss
  int trans_count = bottom[0]->count();
  caffe_sub(
      trans_count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      trans_diff_.mutable_cpu_data());
  Dtype trans_dot = caffe_cpu_dot(trans_count, trans_diff_.cpu_data(), trans_diff_.cpu_data());
  Dtype trans_loss = trans_dot / bottom[0]->num() / Dtype(2);
  // calc rot loss
  int rot_count = bottom[2]->count();
  rot_scale_ = sqrt(caffe_cpu_dot(rot_count, bottom[3]->cpu_data(), bottom[3]->cpu_data()));
  caffe_sub(
      rot_count,
      bottom[2]->cpu_data(),
      bottom[3]->cpu_data(),
      rot_diff_.mutable_cpu_data());
  Dtype rot_dot = caffe_cpu_dot(rot_count, rot_diff_.cpu_data(), rot_diff_.cpu_data());
  Dtype rot_loss = rot_dot / bottom[2]->num() / Dtype(2);
  // calc total loss
  top[0]->mutable_cpu_data()[0] = trans_loss + beta_ / rot_scale_ * rot_loss;
}

template <typename Dtype>
void PoseRegLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // calc diff for trans
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          trans_diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
  // calc diff for rot
  for (int i = 2; i < 4; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 2) ? 1 : -1;
      const Dtype alpha = (i == 2) ? sign * top[0]->cpu_diff()[0] / bottom[i]->num()
          : sign * top[0]->cpu_diff()[0] / bottom[i]->num() / rot_scale_ * beta_;

      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          rot_diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU(PoseRegLayer);
#endif

INSTANTIATE_CLASS(PoseRegLayer);
REGISTER_LAYER_CLASS(PoseReg);

}  // namespace caffe
