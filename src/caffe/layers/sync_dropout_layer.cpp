// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/sync_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SyncDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size());
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void SyncDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (size_t i = 0; i < bottom.size(); i++) {
      CHECK_EQ(bottom[i]->count(), bottom[0]->count());
      top[i]->ReshapeLike(*bottom[i]);
  }
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void SyncDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  unsigned int* mask = NULL;
  if (this->phase_ == TRAIN) {
      mask = rand_vec_.mutable_cpu_data();
      // Create random numbers
      caffe_rng_bernoulli(count, 1. - threshold_, mask);
  }
  for (size_t i = 0; i < bottom.size(); i++) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      if (this->phase_ == TRAIN) {
          for (int j = 0; j < count; ++j) {
              top_data[j] = bottom_data[j] * mask[j] * scale_;
          }
      } else {
          caffe_copy(count, bottom_data, top_data);
      }
  }
}

template <typename Dtype>
void SyncDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const int count = bottom[0]->count();
    const unsigned int* mask = NULL;
    if (this->phase_ == TRAIN) {
        mask = rand_vec_.cpu_data();
    }
    for (size_t i = 0; i < bottom.size(); i++) {
        if (propagate_down[i]) {
            const Dtype* top_diff = top[i]->cpu_diff();
            Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
            if (this->phase_ == TRAIN) {
                for (int j = 0; j < count; ++j) {
                    bottom_diff[j] = top_diff[j] * mask[j] * scale_;
                }
            } else {
                caffe_copy(count, top_diff, bottom_diff);
            }
        }
    }
}


#ifdef CPU_ONLY
STUB_GPU(SyncDropoutLayer);
#endif

INSTANTIATE_CLASS(SyncDropoutLayer);
REGISTER_LAYER_CLASS(SyncDropout);

}  // namespace caffe
