#include <vector>

#include "caffe/layers/rearrange_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Rearrange(const int nthreads, const Dtype* in_data,
                          const bool forward, const int group_num, const int group_id,
                          const int channel_num,    //channel num of a single bottom
                          const int dim,    // size of each channel, height*width
                          Dtype* out_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int channel_id = index / dim;
        const int loc_offset = index % dim;
        const int batch_id = channel_id / channel_num;
        const int channel_offset = channel_id % channel_num;
        const int batch_start = batch_id * (channel_num * dim) * group_num;
        const int channel_start = (group_id + channel_offset * group_num) * dim;
        const int top_index = batch_start + channel_start + loc_offset;

        if (forward) {
            out_data[top_index] = in_data[index];
        } else {
            out_data[index] = in_data[top_index];
        }
    }
}

template <typename Dtype>
void RearrangeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
    Dtype* top_data = top[0]->mutable_gpu_data();
    const bool kForward = true;
    const int group_num = bottom.size();
    const int count = bottom[0]->count();
    const int channel_num = bottom[0]->shape(1);
    const int dim = bottom[0]->shape(2) * bottom[0]->shape(3);
    for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Rearrange<Dtype>
                <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                    count, bottom_data, kForward, group_num, i,
                    channel_num, dim, top_data);
    }
}

template <typename Dtype>
void RearrangeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const bool kForward = false;
    const int group_num = bottom.size();
    const int count = bottom[0]->count();
    const int channel_num = bottom[0]->shape(1);
    const int dim = bottom[0]->shape(2) * bottom[0]->shape(3);
    for (int i = 0; i < bottom.size(); ++i) {
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        Rearrange<Dtype>
                <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                    count, top_diff, kForward, group_num, i,
                    channel_num, dim, bottom_diff);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(RearrangeLayer);

}  // namespace caffe
