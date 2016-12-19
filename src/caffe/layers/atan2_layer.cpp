#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/atan2_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
	void ATan2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	}

template <typename Dtype>
	void ATan2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape = bottom[0]->shape();
		top_shape[1] = 1;
		top[0]->Reshape(top_shape);
	}

template <typename Dtype>
	void ATan2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		const Dtype* input = bottom[0]->cpu_data();
		Dtype* output = top[0]->mutable_cpu_data();

		CHECK(top[0]->count()*2 == bottom[0]->count()) << "Error: in Forward_cpu of ATan2Layer.";

		for(int index = 0; index < top[0]->count(); index++) {
			output[index] = atan2(input[2*index + 1], input[2*index]);
		}
	}

template <typename Dtype>
	void ATan2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {
		const Dtype* output_diff = top[0]->cpu_diff();
		const Dtype* input_data = bottom[0]->cpu_data();
		Dtype* input_diff = bottom[0]->mutable_cpu_diff();

		CHECK(top[0]->count()*2 == bottom[0]->count()) << "Error: in Backward_cpu of ATan2Layer.";

		for(int index = 0; index < top[0]->count(); index++) {
			Dtype x = input_data[2*index];
			Dtype y = input_data[2*index + 1];
			Dtype eps = 1e-8;
			Dtype deno = x*x + y*y + eps;
			input_diff[2*index] = -y/deno * output_diff[index];
			input_diff[2*index+1] = x/deno * output_diff[index];
		}
	}

#ifdef CPU_ONLY
STUB_GPU(ATan2Layer);
#endif

INSTANTIATE_CLASS(ATan2Layer);
REGISTER_LAYER_CLASS(ATan2);

}  // namespace caffe
