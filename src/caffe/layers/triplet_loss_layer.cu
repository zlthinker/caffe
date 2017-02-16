#include <algorithm>
#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
    void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        int count = bottom[0]->count();
        Dtype dis_anchor2pos;
        Dtype dis_anchor2neg;
        Dtype dis_pos2neg;

        // calc diff for anchor2pos, anchor2neg
        caffe_gpu_sub(count, bottom[0]->gpu_data(),
                bottom[1]->gpu_data(),
                diff_anchor2pos_.mutable_gpu_data());
        caffe_gpu_sub(count, bottom[0]->gpu_data(),
                bottom[2]->gpu_data(),
                diff_anchor2neg_.mutable_gpu_data());
        // share pos diff used for bp
        diff_pos_for_bp_.ShareData(diff_anchor2pos_);
        if (intriplet_mining_) {
            caffe_gpu_sub(count, bottom[1]->gpu_data(),
                    bottom[2]->gpu_data(),
                    diff_pos2neg_.mutable_gpu_data());
        } else {
            // if no in-triplet mining, directly share neg diff used for bp
            diff_neg_for_bp_.ShareData(diff_anchor2neg_);
        }

        Dtype loss = 0;
        for (int v = 0; v < batch_size_; ++v) {
            // calc anchor2pos dis
            caffe_gpu_dot(vec_dimension_,
                    diff_anchor2pos_.gpu_data() + v * vec_dimension_,
                    diff_anchor2pos_.gpu_data() + v * vec_dimension_,
                    &dis_anchor2pos);
            vec_loss_.mutable_cpu_data()[v] = alpha_ + dis_anchor2pos;
            // calc anchor2neg dis
            caffe_gpu_dot(vec_dimension_,
                    diff_anchor2neg_.gpu_data() + v * vec_dimension_,
                    diff_anchor2neg_.gpu_data() + v * vec_dimension_,
                    &dis_anchor2neg);
            if (intriplet_mining_) {
                // calc pos2neg ids
                caffe_gpu_dot(vec_dimension_,
                        diff_pos2neg_.gpu_data() + v * vec_dimension_,
                        diff_pos2neg_.gpu_data() + v * vec_dimension_,
                        &dis_pos2neg);
                // if dis(p - n) < dis(a - n) is found, assign the harder one as negative.
                if (dis_pos2neg < dis_anchor2neg) {
                    // with in-triplet mining, neg diff used for bp should be carefully calculated.
                    caffe_copy(vec_dimension_,
                            diff_pos2neg_.gpu_data() + v * vec_dimension_,
                            diff_anchor2neg_.mutable_gpu_data() + v * vec_dimension_);
                    vec_loss_.mutable_cpu_data()[v] -= dis_pos2neg;
                } else {
                    caffe_copy(vec_dimension_,
                            diff_anchor2neg_.gpu_data() + v * vec_dimension_,
                            diff_neg_for_bp_.mutable_gpu_data() + v * vec_dimension_);
                    vec_loss_.mutable_cpu_data()[v] -= dis_anchor2neg;
                }
            } else {
                vec_loss_.mutable_cpu_data()[v] -= dis_anchor2neg;
            }
            vec_loss_.mutable_cpu_data()[v] = std::max(Dtype(0), vec_loss_.cpu_data()[v]);
            loss += vec_loss_.cpu_data()[v];
        }

        loss /= (batch_size_) * Dtype(2);
        top[0]->mutable_cpu_data()[0] = loss;
        if (top.size() == 2) {
            top[1]->ShareData(vec_loss_);
        }
    }

template <typename Dtype>
    void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom) {
        const Dtype scale = top[0]->cpu_diff()[0] / bottom[0]->num();
        const int n = bottom[0]->count();


        caffe_gpu_sub(n, diff_pos_for_bp_.gpu_data(), diff_neg_for_bp_.gpu_data(),
                bottom[0]->mutable_gpu_diff());
        caffe_gpu_scal(n, scale, bottom[0]->mutable_gpu_diff());

        caffe_gpu_scale(n, -scale, diff_pos_for_bp_.gpu_data(),
                bottom[1]->mutable_gpu_diff());

        caffe_gpu_scale(n, scale, diff_neg_for_bp_.gpu_data(),
                bottom[2]->mutable_gpu_diff());

        for (int v = 0; v < batch_size_; ++v) {
            if (vec_loss_.cpu_data()[v] == 0) {
                caffe_gpu_set(vec_dimension_, Dtype(0),
                        bottom[0]->mutable_gpu_diff() + v * vec_dimension_);
                caffe_gpu_set(vec_dimension_, Dtype(0),
                        bottom[1]->mutable_gpu_diff() + v * vec_dimension_);
                caffe_gpu_set(vec_dimension_, Dtype(0),
                        bottom[2]->mutable_gpu_diff() + v * vec_dimension_);
            }
        }
    }

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);

}  // namespace caffe
