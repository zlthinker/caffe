#include <algorithm>
#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
    void TripletLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        // bottom[0]: anchor
        // bottom[1]: positive
        // bottom[2]: negative
        // bottom[3]: label
        CHECK(bottom[0]->shape() == bottom[1]->shape())
            << "Inputs must have the same dimension.";
        CHECK(bottom[0]->shape() == bottom[2]->shape())
            << "Inputs must have the same dimension.";

        diff_anchor2pos_.ReshapeLike(*bottom[0]);
        diff_anchor2neg_.ReshapeLike(*bottom[0]);
        diff_pos2neg_.ReshapeLike(*bottom[0]);

        diff_pos_for_bp_.ReshapeLike(*bottom[0]);
        diff_neg_for_bp_.ReshapeLike(*bottom[0]);

        vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
        top[0]->Reshape(loss_shape);
        batch_size_ = bottom[0]->shape(0);
        vec_dimension_ = bottom[0]->count() / batch_size_;
        if (top.size() == 2) {
            vector<int> per_loss_shape(1);
            per_loss_shape[0] = batch_size_;
            top[1]->Reshape(per_loss_shape);
        }
        vec_loss_.Reshape(batch_size_, 1, 1, 1);
    }

template <typename Dtype>
    void TripletLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        TripletLossParameter param = this->layer_param_.triplet_loss_param();
        CHECK_GE(param.margin_size(), 1);
        for (int i = 0; i < param.margin_size(); i++)
            alpha_.push_back(param.margin(i));
        intriplet_mining_ = param.intriplet_mining();
        if (bottom.size() == 4)
            with_label_ = true;
    }

template <typename Dtype>
    void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        int count = bottom[0]->count();
        const Dtype* label;
        if (with_label_)
            label = bottom[3]->cpu_data();
        Dtype dis_anchor2pos = 0;
        Dtype dis_anchor2neg = 0;
        Dtype dis_pos2neg = 0;

        // calc diff for anchor2pos, anchor2neg
        caffe_sub(count, bottom[0]->cpu_data(),
                bottom[1]->cpu_data(),
                diff_anchor2pos_.mutable_cpu_data());
        caffe_sub(count, bottom[0]->cpu_data(),
                bottom[2]->cpu_data(),
                diff_anchor2neg_.mutable_cpu_data());
        // share pos diff used for bp
        diff_pos_for_bp_.ShareData(diff_anchor2pos_);
        if (intriplet_mining_) {
            caffe_sub(count, bottom[1]->cpu_data(),
                    bottom[2]->cpu_data(),
                    diff_pos2neg_.mutable_cpu_data());
        } else {
            // if no in-triplet mining, directly share neg diff used for bp
            diff_neg_for_bp_.ShareData(diff_anchor2neg_);
        }

        Dtype loss = 0;
        for (int v = 0; v < batch_size_; ++v) {
            // cals anchor2pos dis
            dis_anchor2pos = caffe_cpu_dot(vec_dimension_,
                    diff_anchor2pos_.cpu_data() + v * vec_dimension_,
                    diff_anchor2pos_.cpu_data() + v * vec_dimension_);
            if (with_label_) {
                const int label_value = static_cast<int>(label[v]);
                DCHECK_GE(label_value, 0);
                vec_loss_.mutable_cpu_data()[v] = alpha_[label_value] + dis_anchor2pos;
            } else {
                vec_loss_.mutable_cpu_data()[v] = alpha_[0] + dis_anchor2pos;
            }
            // calc anchor2neg dis
            dis_anchor2neg = caffe_cpu_dot(vec_dimension_,
                    diff_anchor2neg_.cpu_data() + v * vec_dimension_,
                    diff_anchor2neg_.cpu_data() + v * vec_dimension_);
            if (intriplet_mining_) {
                dis_pos2neg = caffe_cpu_dot(vec_dimension_,
                        diff_pos2neg_.cpu_data() + v * vec_dimension_,
                        diff_pos2neg_.cpu_data() + v * vec_dimension_);
                // if dis(p - n) < dis(a - n) is found, assign the harder one as negative.
                if (dis_pos2neg < dis_anchor2neg) {
                    // with in-triplet mining, neg diff used for bp should be carefully calculated.
                    caffe_copy(vec_dimension_,
                            diff_pos2neg_.cpu_data() + v * vec_dimension_,
                            diff_neg_for_bp_.mutable_cpu_data() + v * vec_dimension_);
                    vec_loss_.mutable_cpu_data()[v] -= dis_pos2neg;
                } else {
                    caffe_copy(vec_dimension_,
                            diff_anchor2neg_.cpu_data() + v * vec_dimension_,
                            diff_neg_for_bp_.mutable_cpu_data() + v * vec_dimension_);
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
    void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom) {
        const Dtype scale = top[0]->cpu_diff()[0] / bottom[0]->num();
        const int n = bottom[0]->count();

        caffe_sub(n, diff_pos_for_bp_.cpu_data(), diff_neg_for_bp_.cpu_data(),
                bottom[0]->mutable_cpu_diff());
        caffe_scal(n, scale, bottom[0]->mutable_cpu_diff());

        caffe_cpu_scale(n, -scale, diff_pos_for_bp_.cpu_data(),
                bottom[1]->mutable_cpu_diff());

        caffe_cpu_scale(n, scale, diff_neg_for_bp_.cpu_data(),
                bottom[2]->mutable_cpu_diff());

        for (int v = 0; v < batch_size_; ++v) {
            if (vec_loss_.cpu_data()[v] == 0) {
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
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
