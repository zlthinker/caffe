#include <algorithm>
#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
    void TripletLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        CHECK(bottom[0]->shape() == bottom[1]->shape())
            << "Inputs must have the same dimension.";
        CHECK(bottom[0]->shape() == bottom[2]->shape())
            << "Inputs must have the same dimension.";
        diff_same_class_.ReshapeLike(*bottom[0]);
        diff_diff_class_.ReshapeLike(*bottom[0]);

        vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
        top[0]->Reshape(loss_shape);
        batch_size_ = bottom[0]->shape(0);
        vec_dimension_ = bottom[0]->count() / batch_size_;
        vec_loss_.resize(batch_size_);
        start_idx = floor(batch_size_ * (1.0 - 1.0 / mining_ratio_ ));
    }

template <typename Dtype>
    void TripletLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        alpha_ = this->layer_param_.triplet_loss_param().margin();
        intriplet_mining_ = this->layer_param_.triplet_loss_param().intriplet_mining();
        mining_ratio_ = this->layer_param_.mining_param().mining_ratio();
        if (bottom.size() == 4) {
            semi_hard_ = true;
        }
    }

template <typename Dtype>
    void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        int count = bottom[0]->count();
        Dtype dis_diff_class = 0;

        caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
                diff_same_class_.mutable_cpu_data());
        caffe_sub(count, bottom[0]->cpu_data(), bottom[2]->cpu_data(),
                diff_diff_class_.mutable_cpu_data());
        vector<Dtype> tmp_diff(count);
        if (intriplet_mining_) {
            caffe_sub(count, diff_diff_class_.cpu_data(), diff_same_class_.cpu_data(),
                    &(tmp_diff[0]));
        }

        Dtype loss = 0;
        for (int v = 0; v < batch_size_; ++v) {
            dis_diff_class = caffe_cpu_dot(vec_dimension_,
                    diff_diff_class_.cpu_data() + v * vec_dimension_,
                    diff_diff_class_.cpu_data() + v * vec_dimension_);
            if (intriplet_mining_) {
                Dtype tmp_dis = caffe_cpu_dot(vec_dimension_,
                        &(tmp_diff[v * vec_dimension_]),
                        &(tmp_diff[v * vec_dimension_]));
                // if dis(p - n) < dis(a - n) is found, assign the harder one as negative.
                if (tmp_dis < dis_diff_class) {
                    caffe_copy(vec_dimension_,
                            &(tmp_diff[v * vec_dimension_]),
                            diff_diff_class_.mutable_cpu_data() + v * vec_dimension_);
                    dis_diff_class = tmp_dis;
                }
            }

            vec_loss_[v] =
                caffe_cpu_dot(vec_dimension_,
                        diff_same_class_.cpu_data() + v * vec_dimension_,
                        diff_same_class_.cpu_data() + v * vec_dimension_) -
                dis_diff_class;
            if (semi_hard_) {
                if (static_cast<int>(*(bottom[3]->cpu_data() + v)) == 0)
                    vec_loss_[v] += alpha_;
            } else {
                vec_loss_[v] += alpha_;
            }
            vec_loss_[v] = std::max(Dtype(0), vec_loss_[v]);
        }
        vector<Dtype> tmp_loss = vec_loss_;
        sort (tmp_loss.begin(), tmp_loss.end());
        for (int v = start_idx; v < batch_size_; v++) {
            loss += tmp_loss[v];
        }

        loss /= (batch_size_ - start_idx) * Dtype(2);
        top[0]->mutable_cpu_data()[0] = loss;
    }

template <typename Dtype>
    void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom) {
        const Dtype scale = top[0]->cpu_diff()[0] / bottom[0]->num();
        const int n = bottom[0]->count();

        caffe_sub(n, diff_same_class_.cpu_data(), diff_diff_class_.cpu_data(),
                bottom[0]->mutable_cpu_diff());
        caffe_scal(n, scale, bottom[0]->mutable_cpu_diff());

        caffe_cpu_scale(n, -scale, diff_same_class_.cpu_data(),
                bottom[1]->mutable_cpu_diff());

        caffe_cpu_scale(n, scale, diff_diff_class_.cpu_data(),
                bottom[2]->mutable_cpu_diff());

        for (int v = 0; v < batch_size_; ++v) {
            if (vec_loss_[v] == 0) {
                caffe_set(vec_dimension_, Dtype(0),
                        bottom[0]->mutable_cpu_diff() + v * vec_dimension_);
                caffe_set(vec_dimension_, Dtype(0),
                        bottom[1]->mutable_cpu_diff() + v * vec_dimension_);
                caffe_set(vec_dimension_, Dtype(0),
                        bottom[2]->mutable_cpu_diff() + v * vec_dimension_);
            }
        }
    }

#ifdef CPU_ONLY
// STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
