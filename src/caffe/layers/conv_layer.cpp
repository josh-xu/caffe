#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

// added by xujiang
#ifdef XU_CONV
template <typename Dtype>
void ConvolutionLayer<Dtype>::ComputeBlobMask(float ratio) {
    // LOG(INFO) is cout???
    LOG(INFO) << "conv blob mask" << "\n" ;

    // blobs_[]???
    // count()???
    // CONV_QUNUM???
    int count = this->blobs_[0]->count() ;
    this->masks_.resize(count) ;
    this->indices_.resize(count) ;
    this->centroids_.resize(CONV_QUNUM) ;

    // calculate min max value of weight
    // cpu_data()???
    const Dtype *weight = this->blobs_[0]->cpu_data() ;
    // Dtype min_weight = weight[0] ;
    //Dtype max_weight = weight[0] ;
    vector<Dtype> sort_weight(count) ;

    for (int i = 0; i < count; i++) {
        sort_weight[i] = fabs(weight[i]) ;
    }
    sort(sort_weight.begin(), sort_weight.end()) ;
    //max_weight = sort_weight[count - 1] ;

    std::cout << "sort_weight[0]: " << sort_weight[0] << " " <<
                     "sort_weight[count - 1]: " << sort_weight[count - 1] << "\n" ;
    // what's the usage of index???
    int index = int(count * ratio) ; // int(count * (1 - max_weight)) ;
    Dtype thr ;
    // mutable_cpu_data()???
    Dtype *muweight = this->blobs_[0]->mutable_cpu_data() ;
    float rat = 0 ; // what's the usage of rat???

    if (index > 0) {
        thr = sort_weight[index - 1] ;
        LOG(INFO) << "CONV THR: " << thr << " " << ratio << "\n" ;
        for (int i = 0; i < count; i++) {
            // do the masking!!!
            this->masks_[i] = ((weight[i] >= thr || weight[i] < -thr)? 1 : 0) ;
            muweight[i] *= this->masks_[i] ; // do the prunning by mask!!!
            rat += (1 - this->masks_[i]) ;
        }
    } else {
        for (int i = 0; i < count; i++) {
            // keep unchanged
            this->masks_[i] = ((weight[i] == 0)? 0 : 1) ;
            rat += (1 - this->masks_[i]) ;
        }
    }

    // rat is just used to calculate sparsity???
    LOG(INFO) << "sparsity: " << rat / count << "\n" ;
    // min_weight = sort_weight[index] ; // why min_weight is indexed by index???

    // kmeans_cluster()???
    int nCentroid = CONV_QUNUM ;
    if (nCentroid > count) {
        //std::cout << "@@@ Weird Things Happened!!!\n" ;
        assert(false && "nCentroid > count") ;
        nCentroid = count ;
    }
    std::cout << "nCentroid = CONV_QUNUM: " << nCentroid << "\n" ;
    std::cout << "nWeights = count: " << count << "\n" ;
    kmeans_cluster(this->indices_, this->centroids_, muweight, count,
                       this->masks_, nCentroid, 1000) ;
}
#endif

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // added by xujiang
  #ifdef XU_CONV
  // 01/29/2017, FIND BUG, not only inner_product_layer & conv_layer will call this func!!!
      // will cause segmentation fault!!!
  if (this->masks_.size() != 0) {
      Dtype *muweight = this->blobs_[0]->mutable_cpu_data() ;
      int count = this->blobs_[0]->count() ;

      // 01/29/2017, FIND BUG, not only inner_product_layer & conv_layer will call this func!!!
      //std::cout << "@Forward_cpu() count: " << count << "\n" ;
      //for (int i = 0; i < 16; i++) {
      //    std::cout << this->centroids_[i] << " " ;
      //}
      //std::cout << "\n" ;

      // std::cout << "@Forward_cpu() count: " << count << "\n" ;
      for (int i = 0; i < count; i++) {
          //std::cout << this->masks_[i] << " " ;
          if (this->masks_[i]) {
              // weight sharing!!!
              //std::cout << "Forward_cpu weight sharing iteration " << i << "\n" ;
              //std::cout << this->centroids_[this->indices_[i]] << " " ;
              muweight[i] = this->centroids_[this->indices_[i]] ;
          }
      }
      //std::cout << "\n" ;
  }
  #endif
  // added by xujiang

  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }

        // added by xujiang
        #ifdef XU_CONV
        if (this->masks_.size() != 0) {
            //Dtype *weight_diff = this->blobs_[0]->mutable_cpu_diff() ; // added previously!
            int count = this->blobs_[0]->count() ;

            for (int j = 0; j < count; j++) {
                weight_diff[j] *= this->masks_[j] ; // don't update if mask = 0
            }

            vector<Dtype> tmpDiff(CONV_QUNUM) ;
            vector<int> freq(CONV_QUNUM) ;
            //std::cout << "@print weight_diff[]\n" ;
            for (int j = 0; j < count; j++) {
                // accumulate here
                if (this->masks_[j]) {
                    tmpDiff[this->indices_[j]] += weight_diff[j] ;
                    // added by yuzeng
                    // this->centroids_[this->indices_[j]] -= weight_diff[j];
                    //std::cout << "centroids " << this->centroids_[this->indices_[j]] << "\n" ;
                    freq[this->indices_[j]]++ ;
                }
            }
            //std::cout << "\n" ;

            for (int j = 0; j < count; j++) {
                // mean (average) of gradient diff???
                if (this->masks_[j]) {
                    // weight_diff[j] = tmpDiff[this->indices_[j]] / freq[this->indices_[j]] ;
                    // added by yuzeng
                    this->centroids_[this->indices_[j]] -= LR * weight_diff[j]/freq[this->indices_[j]];
                }
            }
        }

        //for (int j=0; j< CONV_QUNUM; j++){
        //	std::cout << " centroids: " << this->centroids_[j] <<"\n";
        //}
        #endif
        // added by xujiang

        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
