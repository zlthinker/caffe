#include <vector>
#include <cfloat>

#include "caffe/layers/min_max_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RescaleForward(const int nthreads,
    const Dtype* const bottom_data,
    const int dim, const Dtype* const min, const Dtype* const gap,
    Dtype* const top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int channel_id = index / dim;
        Dtype min_val = min[channel_id];
        Dtype gap_val = gap[channel_id];
        top_data[index] = (bottom_data[index] - min_val) / gap_val;
    }
}

template <typename Dtype>
__global__ void RescaleBackward(const int nthreads,
    const Dtype* const top_diff,
    const int dim, const Dtype* const gap,
    Dtype* const bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int channel_id = index / dim;
        Dtype gap_val = gap[channel_id];
        bottom_diff[index] = top_diff[index] * gap_val;
    }
}



template <typename Dtype>
void MinMaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int num = bottom[0]->num();
    int channel = bottom[0]->channels();
    int dim = bottom[0]->height() * bottom[0]->width();
    int count = bottom[0]->count();

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    float delta = this->layer_param_.min_max_param().delta();
    CHECK_GT(delta, 0.0);
    CHECK_LT(delta, 1.0);

    shared_ptr<ConstantFiller<Dtype> > filler;
    FillerParameter filler_param;
    filler_param.set_value(FLT_MAX);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(&this->min_);
    filler_param.set_value(-FLT_MAX);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(&this->max_);
    filler_param.set_value(1.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(&this->gap_);

    // find min and max values within each channel
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channel; c++) {
            const Dtype* feat_map = bottom[0]->cpu_data() + (n * channel + c) * dim;
            Dtype* min_val = this->min_.mutable_cpu_data() + n * channel + c;
            Dtype* max_val = this->max_.mutable_cpu_data() + n * channel + c;
            Dtype* gap_val = this->gap_.mutable_cpu_data() + n * channel + c;
            for (int d = 0; d < dim; d++) {
                Dtype val = feat_map[d];
                if (val > max_val[0]) { max_val[0] = val; }
                if (val < min_val[0]) { min_val[0] = val; }
            }
            if (max_val[0] - min_val[0] > delta) { gap_val[0] = max_val[0] - min_val[0]; }
//            std::cout << c << ", max: " << max_val[0] << ", min: " << min_val[0]
//                      << ", gap: " << gap_val[0] << "\n";
        }
    }

    // [optional] find min and max values across channels
    if (this->layer_param_.min_max_param().across_channels()) {
        std::cout << "[MinMaxLayer] Rescale activations across channels.\n";
        for (int n = 0; n < num; n++) {
            int start_index = n * channel;
            Dtype min_val = this->min_.cpu_data()[start_index];
            Dtype max_val = this->max_.cpu_data()[start_index];
            for (int c = 0; c < channel; c++) {
                Dtype val1 = this->min_.cpu_data()[start_index + c];
                if (val1 < min_val) { min_val = val1; }
                Dtype val2 = this->max_.cpu_data()[start_index + c];
                if (val2 > max_val) { max_val = val2; }
            }
            Dtype gap_val = (max_val - min_val > delta) ? (max_val - min_val) : 1.0;
            for (int c = 0; c < channel; c++) {
                this->min_.mutable_cpu_data()[start_index + c] = min_val;
                this->max_.mutable_cpu_data()[start_index + c] = max_val;
                this->gap_.mutable_cpu_data()[start_index + c] = gap_val;
            }
            std::cout << n << ", max: " << max_val << ", min: " << min_val
                      << ", gap: " << gap_val << "\n";
        }
    }


    RescaleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, dim, this->min_.gpu_data(), this->gap_.gpu_data(), top_data);

}

template <typename Dtype>
void MinMaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    int dim = bottom[0]->height() * bottom[0]->width();

    RescaleBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, dim, this->gap_.gpu_data(), bottom_diff);
}


INSTANTIATE_LAYER_GPU_FUNCS(MinMaxLayer);


}  // namespace caffe
