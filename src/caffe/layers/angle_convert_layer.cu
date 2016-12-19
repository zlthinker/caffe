#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/angle_convert_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
	__global__ void AngleConvertForwardGPU(const int nthreads,
			const Dtype* input, Dtype* output) {

		CUDA_KERNEL_LOOP(index, nthreads) {
			const int offset = index % 6;
			int batchID = index >= 6 ? (index - offset) / 6 : 0;
			Dtype m_cos = cos(input[batchID]);
			Dtype m_sin = sin(input[batchID]);
			switch(offset) {
				case 0:
					output[index] = m_cos;
					break;
				case 1:
					output[index] = m_sin;
					break;
				case 2:
					output[index] = 0;
					break;
				case 3:
					output[index] = -m_sin;
					break;
				case 4:
					output[index] = m_cos;
					break;
				case 5:
					output[index] = 0;
					break;
			}
		}
	}

template <typename Dtype>
	void AngleConvertLayer<Dtype>::Forward_gpu(
			const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* input = bottom[0]->gpu_data();
		Dtype* output = top[0]->mutable_gpu_data();

		CHECK(top[0]->count() == bottom[0]->count()*6) << "Error: in Forward_gpu of AngleConvertLayer.";

		const int nthreads = top[0]->count();
		AngleConvertForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
				nthreads, input, output);
		CUDA_POST_KERNEL_CHECK;
	}

template <typename Dtype>
	__global__ void AngleConvertBackwardGPU(const int nthreads, const Dtype* output_data,
			const Dtype* output_diff, Dtype* input_diff) {

		CUDA_KERNEL_LOOP(index, nthreads) {
			Dtype m_cos = output_data[index*6 + 0];
			Dtype m_sin = output_data[index*6 + 1];
			input_diff[index] = -m_sin * output_diff[index*6 + 0]
				+ m_cos * output_diff[index*6 + 1]
				- m_cos * output_diff[index*6 + 3]
				- m_sin * output_diff[index*6 + 4];
		}
	}

template <typename Dtype>
	void AngleConvertLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* output_diff = top[0]->gpu_diff();
		const Dtype* output_data = top[0]->gpu_data();
		Dtype* input_diff = bottom[0]->mutable_gpu_diff();

		CHECK(top[0]->count() == bottom[0]->count()*6) << "Error: in Backward_gpu of AngleConvertLayer.";

		const int nthreads = bottom[0]->count();
		AngleConvertBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
				nthreads, output_data, output_diff, input_diff);
		CUDA_POST_KERNEL_CHECK;
	}

INSTANTIATE_LAYER_GPU_FUNCS(AngleConvertLayer);

}	// namespace caffe
