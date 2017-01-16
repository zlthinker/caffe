#ifndef TRANSPOSE_LAYER_HPP_
#define TRANSPOSE_LAYER_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class TransposeLayer : public Layer<Dtype> {

public:
	explicit TransposeLayer(const LayerParameter& param)
	: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const  vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const {return "Transpose";}
	virtual inline int ExactNumBottomBlobs() const { return 1;  }
	virtual inline int ExactNumTopBlobs() const { return 1;  }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,  const vector<Blob<Dtype>*>& bottom);

	size_t dim_1;
	size_t dim_2;
};

}  // namespace caffe

#endif  // CAFFE_TRANSPOSE_HPP_
