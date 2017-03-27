#include <vector>

#include "caffe/layers/pose_reg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PoseRegLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // calc trans loss
  int trans_count = bottom[0]->count();
  caffe_sub(
      trans_count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      trans_diff_.mutable_gpu_data());
  Dtype trans_dot;
  caffe_gpu_dot(trans_count, trans_diff_.gpu_data(), trans_diff_.gpu_data(), &trans_dot);
  Dtype trans_loss = trans_dot / bottom[0]->num() / Dtype(2);
  // calc rot loss
  int rot_count = bottom[2]->count();
  Dtype tmp_dot;
  caffe_gpu_dot(rot_count, bottom[3]->gpu_data(), bottom[3]->gpu_data(), &tmp_dot);
  rot_scale_ = sqrt(tmp_dot);
  caffe_sub(
      rot_count,
      bottom[2]->gpu_data(),
      bottom[3]->gpu_data(),
      rot_diff_.mutable_gpu_data());
  Dtype rot_dot;
  caffe_gpu_dot(rot_count, rot_diff_.gpu_data(), rot_diff_.gpu_data(), &rot_dot);
  Dtype rot_loss = rot_dot / bottom[2]->num() / Dtype(2);
  // calc total loss
  top[0]->mutable_gpu_data()[0] = trans_loss + beta_ / rot_scale_ * rot_loss;
}

template <typename Dtype>
void PoseRegLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // calc diff for trans
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->gpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          trans_diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
  // calc diff for rot
  for (int i = 2; i < 4; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 2) ? 1 : -1;
      const Dtype alpha = (i == 2) ? sign * top[0]->gpu_diff()[0] / bottom[i]->num()
          : sign * top[0]->gpu_diff()[0] / bottom[i]->num() / rot_scale_ * beta_;

      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          rot_diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PoseRegLayer);

}  // namespace caffe
