#include <vector>

#include "caffe/layers/rearrange_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RearrangeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    CHECK_GT(bottom.size(), 1);
    vector<int> bottom_shape = bottom[0]->shape();
    for (int i = 1; i < bottom.size(); i++) {
        for (int j = 0; j < bottom_shape.size(); j++) {
            CHECK_EQ(bottom_shape[j], bottom[i]->shape(j));
        }
    }
}

template <typename Dtype>
void RearrangeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
    std::vector<int> top_shape = bottom[0]->shape();
    int group_num = bottom.size();
    top_shape[1] = top_shape[1] * group_num;    // manipulate on channels
    top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RearrangeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // Not implemented
}

template <typename Dtype>
void RearrangeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // Not implemented
}

#ifdef CPU_ONLY
STUB_GPU(RearrangeLayer);
#endif

INSTANTIATE_CLASS(RearrangeLayer);
REGISTER_LAYER_CLASS(Rearrange);

}  // namespace caffe
