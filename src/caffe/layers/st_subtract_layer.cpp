#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/st_subtract_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StSubtractLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void StSubtractLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  top[0]->ReshapeLike(*bottom[0]);
}
template <typename Dtype>
void StSubtractLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data0 = bottom[0]->cpu_data();
  const Dtype* bottom_data1 = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_sub(count, bottom_data1, bottom_data0, top_data);
}

template <typename Dtype>
void StSubtractLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const int count = bottom[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* oppo_top_diff;
  caffe_set(count, Dtype(0), oppo_top_diff);
  caffe_axpy(count, Dtype(-1.0), top_diff, oppo_top_diff);
  Dtype* bottom_diff0 = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff1 = bottom[1]->mutable_gpu_diff();
  caffe_copy(count, oppo_top_diff, bottom_diff0);
  caffe_copy(count, top_diff, bottom_diff1);
}

#ifdef CPU_ONLY
STUB_GPU(StSubtractLayer);
#endif

INSTANTIATE_CLASS(StSubtractLayer);
REGISTER_LAYER_CLASS(StSubtract);

}  // namespace caffe
