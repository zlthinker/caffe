#include <vector>

#include "caffe/layers/st_subtract_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
	__global__ void StSubtractForwardGPU(const int nthreads,
			const Dtype* bottom_data0, const Dtype* bottom_data1, Dtype* top_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			top_data[index] = bottom_data1[index] - bottom_data0[index];
		}
	}

template <typename Dtype>
void StSubtractLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data0 = bottom[0]->gpu_data();
  const Dtype* bottom_data1 = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int nthreads = top[0]->count();
  StSubtractForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
		  nthreads, bottom_data0, bottom_data1, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
	__global__ void StSubtractBackwardGPU(const int nthreads,
			Dtype* bottom_diff0, Dtype* bottom_diff1, const Dtype* top_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			bottom_diff0[index] = -top_diff[index];
			bottom_diff1[index] = top_diff[index];
		}
	}

template <typename Dtype>
void StSubtractLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff0 = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff1 = bottom[1]->mutable_gpu_diff();

  const int nthreads = top[0]->count();
  StSubtractBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
		  nthreads, bottom_diff0, bottom_diff1, top_diff);
  CUDA_POST_KERNEL_CHECK;
  /*const Dtype* test_bottom_diff0 = bottom[0]->cpu_diff();*/
  /*const Dtype* test_bottom_diff1 = bottom[1]->cpu_diff();*/
  /*for(int i = 0; i < nthreads; i++)*/
	  /*LOG(INFO) << test_bottom_diff0[i] << '\t' << test_bottom_diff1[i];*/
}

INSTANTIATE_LAYER_GPU_FUNCS(StSubtractLayer);

}  // namespace caffe
