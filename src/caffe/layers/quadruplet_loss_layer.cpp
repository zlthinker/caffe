#include <algorithm>
#include <vector>

#include "caffe/layers/quadruplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
    void QuadrupletLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        CHECK(bottom[0]->shape() == bottom[1]->shape())
            << "Inputs must have the same dimension.";
        CHECK(bottom[0]->shape() == bottom[2]->shape())
            << "Inputs must have the same dimension.";
        CHECK(bottom[0]->shape() == bottom[3]->shape())
            << "Inputs must have the same dimension.";

        diff_old_.ReshapeLike(*bottom[0]);
        diff_new_.ReshapeLike(*bottom[0]);

        vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
        top[0]->Reshape(loss_shape);
        batch_size_ = bottom[0]->shape(0);
        vec_dimension_ = bottom[0]->count() / batch_size_;
        vec_loss_.resize(batch_size_);
    }

template <typename Dtype>
    void QuadrupletLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        alpha_ = this->layer_param_.threshold_param().threshold();
    }

template <typename Dtype>
    void QuadrupletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        int count = bottom[0]->count();

        caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
                diff_old_.mutable_cpu_data());
        caffe_sub(count, bottom[2]->cpu_data(), bottom[3]->cpu_data(),
                diff_new_.mutable_cpu_data());

        Dtype loss = 0;
        for (int v = 0; v < batch_size_; ++v) {
            loss_old_ = caffe_cpu_dot(vec_dimension_,
                    diff_old_.cpu_data() + v * vec_dimension_,
                    diff_old_.cpu_data() + v * vec_dimension_);
            loss_new_ = caffe_cpu_dot(vec_dimension_,
                    diff_new_.cpu_data() + v * vec_dimension_,
                    diff_new_.cpu_data() + v * vec_dimension_);
            //vec_loss_[v] = loss_new_ + alpha_ * (loss_new_ / loss_old_);
            vec_loss_[v] = alpha_ * ((loss_new_ / loss_old_) - 1);
            vec_loss_[v] = std::max(Dtype(0), vec_loss_[v]);
            loss += vec_loss_[v];
        }

        loss /= batch_size_ * Dtype(2);
        top[0]->mutable_cpu_data()[0] = loss;
    }

template <typename Dtype>
    void QuadrupletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom) {
        const Dtype scale = top[0]->cpu_diff()[0] / bottom[0]->num();
        const int n = bottom[0]->count();

        CHECK_EQ(propagate_down[0], false) << "The first two blob is designed for providing a reference, "
            << "no need to be back-propagated";

        CHECK_EQ(propagate_down[1], false) << "The first two blob is designed for providing a reference, "
            << "no need to be back-propagated";

        caffe_cpu_scale(n, scale*alpha_ / loss_old_, diff_new_.cpu_data(),
                bottom[2]->mutable_cpu_diff());

        caffe_cpu_scale(n, -scale*alpha_ / loss_old_, diff_new_.cpu_data(),
                bottom[3]->mutable_cpu_diff());

        }
#ifdef CPU_ONLY
// STUB_GPU(QuadrupletLossLayer);
#endif

INSTANTIATE_CLASS(QuadrupletLossLayer);
REGISTER_LAYER_CLASS(QuadrupletLoss);

}  // namespace caffe
