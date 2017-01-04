#include <vector>

#include "caffe/layers/rnd_generator_layer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
	void RndGeneratorLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		batch_size_ = this->layer_param_.rnd_generator_param().batch_size();
		dim_ = this->layer_param_.rnd_generator_param().dim();
		scale_ = this->layer_param_.rnd_generator_param().scale();
		shift_ = this->layer_param_.rnd_generator_param().shift();
		range_ = this->layer_param_.rnd_generator_param().range();
		CHECK_GT(batch_size_ * dim_, 0) <<
			"batch_size and dim must be specified and"
			" positive in rnd_generator_param";
		CHECK_GT(range_ , 0) <<
			"range must be specified and"
			" positive in rnd_generator_param";
		top[0]->Reshape(batch_size_, dim_, 1, 1);
		InitRand();
	}

template <typename Dtype>
	void RndGeneratorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		top[0]->Reshape(batch_size_, dim_, 1, 1);
		int count_ = top[0]->count();
		Dtype* top_data = new Dtype[count_];
		for (int i = 0; i < count_; i++) {
			Dtype init = static_cast<Dtype>(Rand(range_));
			Dtype tmp = static_cast<Dtype>((init + shift_) * scale_);
			memcpy(top_data + i, &tmp, sizeof(Dtype));
		}
		top[0]->set_cpu_data(top_data);
	}
template <typename Dtype>
	void RndGeneratorLayer<Dtype>::InitRand() {
		const unsigned int rng_seed = caffe_rng_rand();
		rng_.reset(new Caffe::RNG(rng_seed));
	}

template <typename Dtype>
	int RndGeneratorLayer<Dtype>::Rand(int n) {
		CHECK(rng_);
		CHECK_GT(n, 0);
		caffe::rng_t* rng =
			static_cast<caffe::rng_t*>(rng_->generator());
		return ((*rng)() % n);
	}

INSTANTIATE_CLASS(RndGeneratorLayer);
REGISTER_LAYER_CLASS(RndGenerator);

}  // namespace caffe
