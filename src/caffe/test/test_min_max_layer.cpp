#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/min_max_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MinMaxLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

protected:
    MinMaxLayerTest()
        : blob_bottom_(new Blob<Dtype>(1, 2, 2, 2)),
          blob_top_(new Blob<Dtype>()) {}
    virtual void SetUp() {
        // fill the values
        // channel 1
        this->blob_bottom_->mutable_cpu_data()[0] = 0;
        this->blob_bottom_->mutable_cpu_data()[1] = 1;
        this->blob_bottom_->mutable_cpu_data()[2] = 5;
        this->blob_bottom_->mutable_cpu_data()[3] = 3;
        // channel2
        this->blob_bottom_->mutable_cpu_data()[4] = 1;
        this->blob_bottom_->mutable_cpu_data()[5] = 1.001;
        this->blob_bottom_->mutable_cpu_data()[6] = 1.009;
        this->blob_bottom_->mutable_cpu_data()[7] = 1.005;
        blob_bottom_vec_.push_back(blob_bottom_);
        blob_top_vec_.push_back(blob_top_);
    }

    virtual ~MinMaxLayerTest() {
        delete blob_bottom_;
        delete blob_top_;
    }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MinMaxLayerTest, GPUDevice<float>);

TYPED_TEST(MinMaxLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_min_max_param()->set_delta(0.01);
    layer_param.mutable_min_max_param()->set_across_channels(true);
    MinMaxLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->cpu_data()[0], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[1], 0.2);
    EXPECT_EQ(this->blob_top_->cpu_data()[2], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[3], 0.6);
    EXPECT_EQ(this->blob_top_->cpu_data()[4], 0);
    EXPECT_EQ(this->blob_top_->cpu_data()[5], 0.001);
    EXPECT_EQ(this->blob_top_->cpu_data()[6], 0.009);
    EXPECT_EQ(this->blob_top_->cpu_data()[7], 0.005);
}

//TYPED_TEST(MinMaxLayerTest, TestGradient) {
//    typedef typename TypeParam::Dtype Dtype;
//    LayerParameter layer_param;
//    layer_param.mutable_min_max_param()->set_delta(0.01);
//    MinMaxLayer<Dtype> layer(layer_param);
//    GradientChecker<Dtype> checker(1e-2, 1e-3);
//    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//                                    this->blob_top_vec_);
//    exit(0);
//}

}  // namespace caffe
