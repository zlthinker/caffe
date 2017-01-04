#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/atan2_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
	__global__ void ATan2ForwardGPU(const int nthreads,
			const Dtype* input, Dtype* output) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			output[index] = atan2(input[2*index], input[2*index + 1]);
		}
	}

template <typename Dtype>
	void ATan2Layer<Dtype>::Forward_gpu(
			const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* input = bottom[0]->gpu_data();
		Dtype* output = top[0]->mutable_gpu_data();

		CHECK(top[0]->count()*2 == bottom[0]->count()) << "Error: in Forward_gpu of ATan2Layer.";

		const int nthreads = top[0]->count();
		ATan2ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
				nthreads, input, output);
		CUDA_POST_KERNEL_CHECK;
	}

template <typename Dtype>
	__global__ void ATan2BackwardGPU(const int nthreads, const Dtype* input_data,
			const Dtype* output_diff, Dtype* input_diff) {

		CUDA_KERNEL_LOOP(index, nthreads) {
			Dtype x = input_data[2*index + 1];
			Dtype y = input_data[2*index];
			Dtype eps = 1e-8;
			Dtype deno = x*x + y*y + eps;
			input_diff[2*index] = x/deno * output_diff[index];
			input_diff[2*index+1] = -y/deno * output_diff[index];
		}
	}

template <typename Dtype>
	void ATan2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* output_diff = top[0]->gpu_diff();
		const Dtype* input_data = bottom[0]->gpu_data();
		Dtype* input_diff = bottom[0]->mutable_gpu_diff();

		CHECK(top[0]->count()*2 == bottom[0]->count()) << "Error: in Backward_gpu of ATan2Layer.";

		const int nthreads = top[0]->count();
		ATan2BackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
				nthreads, input_data, output_diff, input_diff);
		CUDA_POST_KERNEL_CHECK;
//		const Dtype* test_bottom = bottom[0]->cpu_data();
//		const Dtype* test_top = top[0]->cpu_data();
//		const Dtype* diff_bottom = bottom[0]->cpu_diff();
//		const Dtype* diff_top = top[0]->cpu_diff();
//		for(size_t i = 0; i < top[0]->count(); i++) {
//			LOG(INFO) << "bottom: " << test_bottom[2*i] << '\t' << test_bottom[2*i + 1];
//			LOG(INFO) << "top: " << test_top[i];
//			LOG(INFO) << "diff bottom: " << diff_bottom[2*i] << '\t' << diff_bottom[2*i + 1];
//			LOG(INFO) << "top bottom: " << diff_top[i] << std::endl;
//		}
	}

INSTANTIATE_LAYER_GPU_FUNCS(ATan2Layer);

}	// namespace caffe
