#include <vector>
#include <limits>
#include <cfloat>

#include "caffe/layers/min_max_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MinMaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(bottom[0]->shape());
    min_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    max_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    gap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);

}

template <typename Dtype>
void MinMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int num = bottom[0]->num();
    int channel = bottom[0]->channels();
    int dim = bottom[0]->height() * bottom[0]->width();

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

    float delta = this->layer_param_.min_max_param().delta();
    CHECK_GT(delta, 0.0);
    CHECK_LT(delta, 1.0);

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
        }
    }

    // rescale activations within each channel  y = (y - min)/(max - min)
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channel; c++) {
            Dtype* bottom_data = bottom[0]->mutable_cpu_data() + (n * channel + c) * dim;
            Dtype* top_data = top[0]->mutable_cpu_data() + (n * channel + c) * dim;
            Dtype min_val = this->min_.cpu_data()[n * channel + c];
            Dtype gap_val = this->gap_.cpu_data()[n * channel + c];
            for (int d = 0; d < dim; d++) {
                top_data[d] = (bottom_data[d] - min_val) / gap_val;
            }
        }
    }

}

template <typename Dtype>
void MinMaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    int num = bottom[0]->num();
    int channel = bottom[0]->channels();
    int dim = bottom[0]->height() * bottom[0]->width();

    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
//    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channel; c++) {
            Dtype gap_val = this->gap_.cpu_data()[n * channel + c];
            int index = (n * channel + c) * dim;
            // y = a*X + y
            caffe_axpy(dim, 1.0 / gap_val, top_diff + index, bottom_diff + index);
        }
    }
}


#ifdef CPU_ONLY
STUB_GPU(MinMaxLayer);
#endif

INSTANTIATE_CLASS(MinMaxLayer);
REGISTER_LAYER_CLASS(MinMax);

}  // namespace caffe
