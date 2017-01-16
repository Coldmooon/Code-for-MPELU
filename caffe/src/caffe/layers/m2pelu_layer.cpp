#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/m2pelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void M2PELULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  M2PELUParameter m2pelu_param = this->layer_param().m2pelu_param();
  int channels = bottom[0]->channels();
  channel_shared_ = m2pelu_param.channel_shared();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
      this->blobs_[1].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
      this->blobs_[1].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > filler;
    if (m2pelu_param.has_alpha_filler()) {
      filler.reset(GetFiller<Dtype>(m2pelu_param.alpha_filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
    if (m2pelu_param.has_beta_filler()) {
      filler.reset(GetFiller<Dtype>(m2pelu_param.beta_filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.0);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[1].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Alpha size is inconsistent with prototxt config";
    CHECK_EQ(this->blobs_[1]->count(), 1)
        << "Beta  size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Alpha size is inconsistent with prototxt config";
    CHECK_EQ(this->blobs_[1]->count(), channels)
        << "Beta  size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_alpha.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_beta.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void M2PELULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
  // CHECK_NE(bottom[0], top[0]) << "***In-place computation is not allowed, because top will be used in the backward pass.***";
}

template <typename Dtype>
void M2PELULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  const Dtype* alpha = this->blobs_[0]->cpu_data();
  const Dtype* beta  = this->blobs_[1]->cpu_data();

  // CHECK_NE(bottom[0], top[0]) << "***In-place computation is not allowed, because top will be used in the backward pass.***";
  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + alpha[c] * ( exp(beta[c]*std::min(bottom_data[i], Dtype(0))) - 1);
  }
}

template <typename Dtype>
void M2PELULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();

  const Dtype* alpha = this->blobs_[0]->cpu_data();
  const Dtype* beta  = this->blobs_[1]->cpu_data();

  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // CHECK_NE(bottom[0], top[0]) 
  					// << "***In-place computation is not allowed, because top will be used in the backward pass.***";
  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  // Propagte to param alpha
  if (this->param_propagate_down_[0]) {
    Dtype* alpha_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* beta_diff  = this->blobs_[1]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      // CHECK_NE(alpha, 0) << "alpha can not be zero in the backward pass in M2PELU"
      alpha_diff[c] += top_diff[i] * ( (bottom_data[i] <= 0)? (exp(beta[c]*bottom_data[i]) - 1 ): Dtype(0)  );
      // alpha_diff[c] += top_diff[i] * (bottom_data[i] <= 0) * (exp(beta[c]*bottom_data[i] + gamma[c]) - exp(gamma[c]) );
      beta_diff[c]  += top_diff[i] * ( bottom_data[i]*( top_data[i] + alpha[c] ) * (bottom_data[i] <= 0) );
      // beta_diff[c]  += top_diff[i] * alpha[c] * bottom_data[i] * exp(beta[c] * bottom_data[i] + gamma[c]) * (bottom_data[i] <= 0);
      // if (bottom_data[i] == 0) CHECK_EQ(top_data[i], 0) << "gamma diff: top is not zero when bottom is.";
      bottom_diff[i] = top_diff[i] * ( (bottom_data[i] > 0) + beta[c]*( top_data[i] + alpha[c]) * (bottom_data[i] <= 0) );
    }
  }

//   // Propagate to bottom
//   if (propagate_down[0]) {
//     Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
//     for (int i = 0; i < count; ++i) {
//       int c = (i / dim) % channels / div_factor;
//       bottom_diff[i] = top_diff[i] * ( (bottom_data[i] > 0) + beta[c]*( top_data[i] + alpha[c]*exp(gamma[c])) * (bottom_data[i] <= 0) );
//       // bottom_diff[i] = top_diff[i] * ( (bottom_data[i] > 0) + alpha[c] * beta[c] * exp(beta[c]*bottom_data[i] + gamma[c])*(bottom_data[i] <= 0) );
//       // if (bottom_data[i] == 0) CHECK_EQ(top_data[i], 0) << "input diff: top is not zero when bottom is.";
//     }
//   }
}


#ifdef CPU_ONLY
STUB_GPU(M2PELULayer);
#endif

INSTANTIATE_CLASS(M2PELULayer);
REGISTER_LAYER_CLASS(M2PELU);

}  // namespace caffe
