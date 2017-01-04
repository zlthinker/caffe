#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/angle_convert_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
	void AngleConvertLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		scale_ = this->layer_param_.angle_convert_param().scale();
	}

template <typename Dtype>
	void AngleConvertLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape = bottom[0]->shape();
		top_shape[1] = 6;
		top[0]->Reshape(top_shape);
	}

template <typename Dtype>
	void AngleConvertLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		const Dtype* input = bottom[0]->cpu_data();
		Dtype* output = top[0]->mutable_cpu_data();

		CHECK(top[0]->count() == bottom[0]->count()*6) << "Error: in Forward_cpu of AngleConvertLayer.";

		for(int index = 0; index < top[0]->count(); index++) {
			const int offset = index % 6;
			int batchID = index >= 6 ? (index - offset) / 6 : 0;
			Dtype m_cos = cos(input[batchID]);
			Dtype m_sin = sin(input[batchID]);
			switch(offset) {
				case 0:
					output[index] = scale_ * m_cos;
					break;
				case 1:
					output[index] = -scale_ * m_sin;
					break;
				case 2:
					output[index] = 0;
					break;
				case 3:
					output[index] = scale_ * m_sin;
					break;
				case 4:
					output[index] = scale_ * m_cos;
					break;
				case 5:
					output[index] = 0;
					break;
			}
		}
	}

template <typename Dtype>
	void AngleConvertLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {
		const Dtype* output_diff = top[0]->cpu_diff();
		const Dtype* output_data = top[0]->cpu_data();
		Dtype* input_diff = bottom[0]->mutable_cpu_diff();

		CHECK(top[0]->count() == bottom[0]->count()*6) << "Error: in Backward_cpu of AngleConvertLayer.";

		for(int index = 0; index < bottom[0]->count(); index++) {
			Dtype m_cos = output_data[index*6 + 0];
			Dtype m_sin = output_data[index*6 + 1];
			input_diff[index] = -scale_ * m_sin * output_diff[index*6 + 0]
				- scale_ * m_cos * output_diff[index*6 + 1]
				+ scale_ * m_cos * output_diff[index*6 + 3]
				- scale_ * m_sin * output_diff[index*6 + 4];
		}
	}

#ifdef CPU_ONLY
STUB_GPU(AngleConvertLayer);
#endif

INSTANTIATE_CLASS(AngleConvertLayer);
REGISTER_LAYER_CLASS(AngleConvert);

}  // namespace caffe
