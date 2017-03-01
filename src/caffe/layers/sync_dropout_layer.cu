#include <vector>

#include "caffe/layers/sync_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SyncDropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
void SyncDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const int count = bottom[0]->count();
    unsigned int* mask = NULL;
    if (this->phase_ == TRAIN) {
        mask = static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
        caffe_gpu_rng_uniform(count, mask);
    }
    for (size_t i = 0; i < bottom.size(); i++) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = top[i]->mutable_gpu_data();
        if (this->phase_ == TRAIN) {
            // set thresholds
            // NOLINT_NEXT_LINE(whitespace/operators)
            SyncDropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                    count, bottom_data, mask, uint_thres_, scale_, top_data);
            CUDA_POST_KERNEL_CHECK;
        } else {
            caffe_copy(count, bottom_data, top_data);
        }
    }
}

template <typename Dtype>
__global__ void SyncDropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
void SyncDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const int count = bottom[0]->count();
    const unsigned int* mask = NULL;
    if (this->phase_ == TRAIN) {
        mask = static_cast<const unsigned int*>(rand_vec_.gpu_data());
    }
    for (size_t i = 0; i < bottom.size(); i++) {
        if (propagate_down[i]) {
            const Dtype* top_diff = top[i]->gpu_diff();
            Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
            if (this->phase_ == TRAIN) {
                // NOLINT_NEXT_LINE(whitespace/operators)
                SyncDropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
                    CAFFE_CUDA_NUM_THREADS>>>(
                            count, top_diff, mask, uint_thres_, scale_, bottom_diff);
                CUDA_POST_KERNEL_CHECK;
            } else {
                caffe_copy(count, top_diff, bottom_diff);
            }
        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(SyncDropoutLayer);

}  // namespace caffe
