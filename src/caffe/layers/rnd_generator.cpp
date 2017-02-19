#include <vector>

#include "caffe/layers/rnd_generator_layer.hpp"
#include "caffe/util/rng.hpp"

#define PI 3.14159265
namespace caffe {

template <typename Dtype>
    void RndGeneratorLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        batch_size_ = this->layer_param_.rnd_generator_param().batch_size();
        CHECK_GT(batch_size_, 0) <<
            "batch_size must be specified and"
            " positive in rnd_generator_param";
        if (this->layer_param_.rnd_generator_param().rnd_method()
            == RndGeneratorParameter_RndMethod_NAIVE) {
            // For NAIVE method
            dim_ = this->layer_param_.rnd_generator_param().dim();
            scale_ = this->layer_param_.rnd_generator_param().scale();
            shift_ = this->layer_param_.rnd_generator_param().shift();
            range_ = this->layer_param_.rnd_generator_param().range();
            CHECK_GT(dim_ , 0) <<
                "dim must be specified and"
                " positive in rnd_generator_param";
            CHECK_GT(range_ , 0) <<
                "range must be specified and"
                " positive in rnd_generator_param";
            top[0]->Reshape(batch_size_, dim_, 1, 1);
        } else if (this->layer_param_.rnd_generator_param().rnd_method()
            == RndGeneratorParameter_RndMethod_HARD_PATCH_MINING ||
            this->layer_param_.rnd_generator_param().rnd_method()
            == RndGeneratorParameter_RndMethod_TRANS_AUG) {
            top[0]->Reshape(batch_size_, 6, 1, 1);
        }
        InitRand();
    }

template <typename Dtype>
    void RndGeneratorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        if (this->layer_param_.rnd_generator_param().rnd_method()
            == RndGeneratorParameter_RndMethod_NAIVE) {
            top[0]->Reshape(batch_size_, dim_, 1, 1);
            int count_ = top[0]->count();
            for (int i = 0; i < count_; i++) {
                Dtype init = static_cast<Dtype>(Rand(range_));
                Dtype tmp = static_cast<Dtype>((init + shift_) * scale_);
                *(top[0]->mutable_cpu_data() + i) = tmp;
            }
        } else if (this->layer_param_.rnd_generator_param().rnd_method()
                == RndGeneratorParameter_RndMethod_HARD_PATCH_MINING) {
            top[0]->Reshape(batch_size_, 6, 1, 1);
            for (int i = 0; i < batch_size_; i++) {
                Dtype crop_param[6] = {0.5, .0, .0, .0, 0.5, .0};
                // init the x/y direction
                int x_or_y = Rand(2);
                // init the pos/neg direction
                int pos_or_neg = Rand(2);
                // init the perturbation
                Dtype perturb = static_cast<Dtype>((static_cast<Dtype>(Rand(100)) - 50.0) * 0.01);
                if (x_or_y == 0) {
                    if (pos_or_neg == 0)
                        crop_param[2] = -0.5;
                    else if (pos_or_neg == 1)
                        crop_param[2] = 0.5;
                    crop_param[5] = perturb;
                } else if (x_or_y == 1) {
                    if (pos_or_neg == 0)
                        crop_param[5] = -0.5;
                    else if (pos_or_neg == 1)
                        crop_param[5] = 0.5;
                    crop_param[2] = perturb;
                }
                caffe_copy(6, crop_param, top[0]->mutable_cpu_data() + i*6);
            }
        } else if (this->layer_param_.rnd_generator_param().rnd_method()
                == RndGeneratorParameter_RndMethod_TRANS_AUG) {
            for (int i = 0; i < batch_size_; i++) {
                Dtype crop_scale = this->layer_param_.rnd_generator_param().crop_scale();
                Dtype trans_param[6] = {crop_scale, .0, .0, .0, crop_scale, .0};
                // note ini this implemention, three transformation components are independent
                // rotate
                if (this->layer_param_.rnd_generator_param().rotate() == true) {
                    Dtype rotate_deg = this->layer_param_.rnd_generator_param().rotate_deg();
                    rotate_deg = rotate_deg - Rand(rotate_deg * 2);
                    Dtype m_cos = cos(rotate_deg*PI/180);
                    Dtype m_sin = sin(rotate_deg*PI/180);
                    trans_param[0] = m_cos * crop_scale;
                    trans_param[1] = -m_sin * crop_scale;
                    trans_param[3] = m_sin * crop_scale;
                    trans_param[4] = m_cos * crop_scale;
                }
                // naive_rotate
                if (this->layer_param_.rnd_generator_param().naive_rotate() == true) {
                    Dtype naive_rotation_pool[6] = {0.0, 0.0, 0.0, -90.0, 90.0, 180.0};
                    int idx = Rand(6);
                    Dtype m_cos = cos(naive_rotation_pool[idx]*PI/180);
                    Dtype m_sin = sin(naive_rotation_pool[idx]*PI/180);
                    trans_param[0] = m_cos * crop_scale;
                    trans_param[1] = -m_sin * crop_scale;
                    trans_param[3] = m_sin * crop_scale;
                    trans_param[4] = m_cos * crop_scale;
                }
                // translation
                if (this->layer_param_.rnd_generator_param().translate() == true) {
                    Dtype trans_x = this->layer_param_.rnd_generator_param().trans_x();
                    trans_x = (trans_x - Rand(trans_x * 2)) / 100.0;
                    Dtype trans_y = this->layer_param_.rnd_generator_param().trans_y();
                    trans_y = (trans_y - Rand(trans_y * 2)) / 100.0;
                    trans_param[2] += trans_x;
                    trans_param[5] += trans_y;
                }
                // zoom
                if (this->layer_param_.rnd_generator_param().zoom() == true) {
                    Dtype zoom_scale = this->layer_param_.rnd_generator_param().zoom_scale();
                    zoom_scale = (100 + zoom_scale - Rand(zoom_scale * 2)) / 100.0;
                    trans_param[0] *= zoom_scale;
                    trans_param[1] *= zoom_scale;
                    trans_param[3] *= zoom_scale;
                    trans_param[4] *= zoom_scale;
                }
                caffe_copy(6, trans_param, top[0]->mutable_cpu_data() + i*6);
            }
        }
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
