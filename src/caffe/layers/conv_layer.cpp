#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>

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
    std::ofstream FD;
    //FD.open("/Users/xujiang/technologies/caffe/output_CONV.log", std::ios_base::app);
    FD.open("/Users/xujiang/technologies/caffe/output_CONV.log", std::fstream::app);
    FD << "conv blob mask" << std::endl ;

    // blobs_[]???
    // count()???
    // CONV_QUNUM???
    //this->msk_no = 3; // -- del
    
    int count = this->blobs_[0]->count() ;
    /*int mask_no = 5;
    int count = this->blobs_[0]->count() ;
    this->masks_.resize(count) ;
    this->masks1_.resize(count) ;  
    this->masks2_.resize(count) ;  
    this->masks3_.resize(count) ;
    this->masks4_.resize(count) ;*/
    // added by xujiang, 02/08/2017
    // Fix BUG here, 02/09/2017, should push ptr, not masks_ itself, otherwise the masks_ will be copied...
    this->mask_vec_.push_back(&(this->masks_));
    #ifdef ONE_BIT
        this->mask_vec_.push_back(&(this->masks1_));
        this->mask_vec_.push_back(&(this->masks2_));
        this->mask_vec_.push_back(&(this->masks3_));
        this->mask_vec_.push_back(&(this->masks4_));
        this->mask_vec_.push_back(&(this->masks5_));
    #endif
    #ifdef TWO_BIT
        this->mask_vec2b_.push_back(&(this->masks5p6_));
        this->mask_vec2b_.push_back(&(this->masks5p7_));
    #endif
    for (vector< vector<int>* >::iterator it = this->mask_vec_.begin();
             it != this->mask_vec_.end(); it++) {
        (*it)->resize(count);
    }
    #ifdef TWO_BIT
        for (vector< vector<int>* >::iterator it = this->mask_vec2b_.begin();
                 it != this->mask_vec2b_.end(); it++) {
            (*it)->resize(count);
        }
    #endif

    this->masks_all.resize(count) ;
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

    FD << "sort_weight[0]: " << sort_weight[0] << " " <<
                     "sort_weight[count - 1]: " << sort_weight[count - 1] << "\n" ;
    // what's the usage of index???
    int index = int(count * ratio) ; // int(count * (1 - max_weight)) ;
    //Dtype thr[0] ;
    vector<Dtype> thr;
    vector<Dtype> thr2b;
    //thr.resize(mask_no);
    //for (int i = 0; i < this->msk_no; i++){  // set the thr --yuzeng -- del
    //FD << "The thrs :" << std::endl;
    /*for (int i = 0; i < mask_no; i++){  // --del
      thr.push_back(0.2); 
      //FD << thr[i] << "  ";
    }*/
    // added by xujiang, 02/08/2017
    for (vector< vector<int>* >::iterator it = this->mask_vec_.begin();
             it != this->mask_vec_.end(); it++) {
        thr.push_back(0.2);
    }
    #ifdef TWO_BIT
        for (vector< vector<int>* >::iterator it = this->mask_vec2b_.begin();
                 it != this->mask_vec2b_.end(); it++) {
            thr2b.push_back(0.05);
        }
    #endif

    //FD << std::endl;

    // mutable_cpu_data()???
    Dtype *muweight = this->blobs_[0]->mutable_cpu_data() ;
    //float rat = 0 ; // what's the usage of rat???
    vector<float> prune;
    vector<float> prune2b;
    //prune.resize(mask_no);
    //float prune[0] = 0;
    //for (int i = 0; i < this->msk_no; i++){  // set the prune number --yuzeng -- del
    /*for (int i = 0; i < mask_no; i++){  //-- del
      prune.push_back(0); 
    }*/
    // added by xujiang, 02/08/2017
    for (vector< vector<int>* >::iterator it = this->mask_vec_.begin();
             it != this->mask_vec_.end(); it++) {
        prune.push_back(0);
    }
    #ifdef TWO_BIT
        for (vector< vector<int>* >::iterator it = this->mask_vec2b_.begin();
                 it != this->mask_vec2b_.end(); it++) {
            prune2b.push_back(0);
        }
    #endif

    //FD << "masks: " << std::endl;
    if (index > 0) {
        thr[0] = sort_weight[index - 1] ;
        //FD << "CONV THR: " << thr[0] << " " << ratio << std::endl ;
        std::cout << "CONV THR: " << thr[0] << " " << ratio << std::endl ;
        for (int i = 0; i < count; i++) {
            // do the masking!!!
            this->masks_[i] = ((weight[i] >= thr[0] || weight[i] < -thr[0])? 1 : 0) ;
            //FD << this->masks_[i];
            muweight[i] *= this->masks_[i] ; // do the prunning by mask!!!
            prune[0] += (1 - this->masks_[i]) ;
        }
    } else {
        for (int i = 0; i < count; i++) {
            // keep unchanged
            this->masks_[i] = ((weight[i] == 0)? 0 : 1) ;
            prune[0] += (1 - this->masks_[i]) ;
        }
    }

    // rat is just used to calculate sparsity???
    FD << "percent of 0: " << prune[0] / count << std::endl ;
    FD << "prune[0]: " << prune[0] << std::endl ;
    // min_weight = sort_weight[index] ; // why min_weight is indexed by index???

    #ifdef ONE_BIT
        // added by xujiang, 02/08/2017
        vector<int> pow_vec_;
        pow_vec_.push_back(0); // not used!!! used to align index...
        pow_vec_.push_back(1);
        pow_vec_.push_back(2);
        pow_vec_.push_back(3);
        pow_vec_.push_back(4);
        pow_vec_.push_back(5);
        int mask_iter_count_ = 1;
        for (vector< vector<int>* >::iterator it = (this->mask_vec_.begin() + 1); // skip mask0
                 it != this->mask_vec_.end(); it++, mask_iter_count_++) {
            for (int i = 0; i < count; i++) {
                float val = 1/pow(2, pow_vec_[mask_iter_count_]); // 1 2 3 4 5
                float set_val = (weight[i] > 0)? val : (-1) *val;
                //this->masks1_[i] = ((fabs(weight[i]) >= (1-thr[1])*val && fabs(weight[i]) <= (1+thr[1])*val)? 0 : 1) ;
                (*(*it))[i] = ((fabs(weight[i]) >= (1-thr[mask_iter_count_])*val && fabs(weight[i]) <= (1+thr[mask_iter_count_])*val)? 0 : 1) ;
                //muweight[i] = ((this->masks1_[i] == 0) ? set_val :  muweight[i]) ;
                muweight[i] = (((*(*it))[i] == 0 && this->masks_[i] != 0) ? set_val :  muweight[i]) ;
                //prune[1] += (1 - this->masks1_[i]);
                prune[mask_iter_count_] += (1 - (*(*it))[i]);
            }
            //FD << "percent of 1/2: " << prune[1] / count << std::endl ;
            //FD << "prune[1]: " << prune[1] << std::endl ;
            FD << "percent of 1/" << pow(2, pow_vec_[mask_iter_count_]) << ": " << prune[mask_iter_count_] / count << std::endl ;
            FD << "prune[" << mask_iter_count_ << "]: " << prune[mask_iter_count_] << std::endl ;
        }
    #endif
    #ifdef TWO_BIT
        vector<float> pow_vec2b_;
        pow_vec2b_.push_back(1/pow(2, 5) + 1/pow(2, 6));
        pow_vec2b_.push_back(1/pow(2, 5) + 1/pow(2, 7));
        int mask_iter_count2b_ = 0;
        for (vector< vector<int>* >::iterator it = this->mask_vec2b_.begin();
                 it != this->mask_vec2b_.end(); it++, mask_iter_count2b_++) {
            for (int i = 0; i < count; i++) {
                float val = pow_vec2b_[mask_iter_count2b_]; // 5p6 5p7
                float set_val = (weight[i] > 0)? val : (-1) *val;
                (*(*it))[i] = ((fabs(weight[i]) >= (1-thr2b[mask_iter_count2b_])*val && fabs(weight[i]) <= (1+thr2b[mask_iter_count2b_])*val)? 0 : 1) ;
                muweight[i] = (((*(*it))[i] == 0 && this->masks_[i] != 0) ? set_val :  muweight[i]) ;
                prune2b[mask_iter_count2b_] += (1 - (*(*it))[i]);
            }
            FD << "percent of " << pow_vec2b_[mask_iter_count2b_] << ": " << prune2b[mask_iter_count2b_] / count << std::endl ;
            FD << "prune2b[" << mask_iter_count2b_ << "]: " << prune2b[mask_iter_count2b_] << std::endl ;
        }
    #endif

    /*//FD << "masks1: " << std::endl;
    for(int i = 0; i < count; i++) {
      float val = 1/pow(2, 1);
      float set_val = (weight[i] > 0)? val : (-1) *val; 
      this->masks1_[i] = ((fabs(weight[i]) >= (1-thr[1])*val && fabs(weight[i]) <= (1+thr[1])*val)? 0 : 1) ;
      //FD << this->masks1_[i];
      muweight[i] = ((this->masks1_[i] == 0) ? set_val :  muweight[i]) ;
      prune[1] += (1 - this->masks1_[i]);
    }

    FD << "percent of 1/2: " << prune[1] / count << std::endl ;
    FD << "prune[1]: " << prune[1] << std::endl ;

    //FD << "masks2: " << std::endl;
    for(int i = 0; i < count; i++) {
      float val = 1/pow(2, 2);
      float set_val = (weight[i] > 0)? val : (-1) *val;
      this->masks2_[i] = ((fabs(weight[i]) >= (1-thr[2])*val && fabs(weight[i]) <= (1+thr[2])*val)? 0 : 1) ;
      //FD << this->masks2_[i];
      muweight[i] = ((this->masks2_[i] == 0) ? set_val :  muweight[i]) ;
      prune[2] += (1 - this->masks2_[i]);
    }

    FD << "percent of 1/4: " << prune[2] / count << std::endl ;
    FD << "prune[2]: " << prune[2] << std::endl ;

    //FD << "masks3: " << std::endl;
    for(int i = 0; i < count; i++) {
      float val = 1/pow(2, 3);
      float set_val = (weight[i] > 0)? val : (-1) *val;
      this->masks3_[i] = ((fabs(weight[i]) >= (1-thr[3])*val && fabs(weight[i]) <= (1+thr[3])*val)? 0 : 1) ;
      //FD << this->masks3_[i];
      muweight[i] = ((this->masks3_[i] == 0) ? set_val :  muweight[i]) ;
      prune[3] += (1 - this->masks3_[i]);
    }

    FD << "percent of 1/8: " << prune[3] / count << std::endl ;
    FD << "prune[3]: " << prune[3] << std::endl ;

    //FD << "masks4: " << std::endl;
    for(int i = 0; i < count; i++) {
      float val = 1/pow(2, 4);
      float set_val = (weight[i] > 0)? val : (-1) *val;
      this->masks4_[i] = ((fabs(weight[i]) >= (1-thr[4])*val && fabs(weight[i]) <= (1+thr[4])*val)? 0 : 1) ;
      //FD << this->masks4_[i];
      muweight[i] = ((this->masks4_[i] == 0) ? set_val :  muweight[i]) ;
      prune[4] += (1 - this->masks4_[i]);
    }

    FD << "percent of 1/16: " << prune[4] / count << std::endl ;
    FD << "prune[4]: " << prune[4] << std::endl ;*/
    
    // initialize the masks_all;

    for (int i = 0; i < count; i++) {
      this->masks_all.push_back(1);
    }
    
    //FD << "masks_all :" <<std::endl;
    for (int i = 0; i < count; i++){
      //for(int j = 0; j < mask_no; j++)
      //this->masks_all[i] = this->masks_[i] & this->masks1_[i] & this->masks2_[i] & this->masks3_[i] & this->masks4_[i];
      // added by xujiang, 02/08/2017
      this->masks_all[i] = this->masks_[i];
      #ifdef ONE_BIT
          for (vector< vector<int>* >::iterator it = (this->mask_vec_.begin() + 1); // skip mask0
                               it != this->mask_vec_.end(); it++) {
              this->masks_all[i] = this->masks_all[i] & (*(*it))[i];
          }
      #endif
      #ifdef TWO_BIT
          for (vector< vector<int>* >::iterator it = this->mask_vec2b_.begin();
                               it != this->mask_vec2b_.end(); it++) {
              this->masks_all[i] = this->masks_all[i] & (*(*it))[i];
          }
      #endif
      //FD << this->masks_all[i];
    }

    // kmeans_cluster()???
    int nCentroid = CONV_QUNUM ;
    if (nCentroid > count) {
        //FD << "@@@ Weird Things Happened!!!\n" ;
        assert(false && "nCentroid > count") ;
        nCentroid = count ;
    }
    FD << "nCentroid = CONV_QUNUM: " << nCentroid << "\n" ;
    FD << "nWeights = count: " << count << "\n" ;
    kmeans_cluster(this->indices_, this->centroids_, muweight, count,
                       this->masks_all, nCentroid, 1000) ;
    // added by yuzeng
    float sparsity_post = 0;
    for (int i = 0; i < count; i++ ){
      //std::cout << this->indices_[i];
      if (this->indices_[i] == -1)
        sparsity_post += 1;
    }
    FD << "sparsity after kmeans " << sparsity_post / count << std::endl;
    FD << "prune all: " << sparsity_post << std::endl ;
    FD << "################# The end of conv layer data ####################" << std::endl;
    FD.close();
}
#endif

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // added by xujiang
  #ifdef XU_CONV
  // 01/29/2017, FIND BUG, not only inner_product_layer & conv_layer will call this func!!!
      // will cause segmentation fault!!!
  if (this->masks_all.size() != 0) {
      Dtype *muweight = this->blobs_[0]->mutable_cpu_data() ;
      int count = this->blobs_[0]->count() ;

      // 01/29/2017, FIND BUG, not only inner_product_layer & conv_layer will call this func!!!
      //FD << "@Forward_cpu() count: " << count << "\n" ;
      //for (int i = 0; i < 16; i++) {
      //    FD << this->centroids_[i] << " " ;
      //}
      //FD << "\n" ;

      // FD << "@Forward_cpu() count: " << count << "\n" ;
      for (int i = 0; i < count; i++) {
          //FD << this->masks_[i] << " " ;
          if (this->masks_all[i]) {
              // weight sharing!!!
              //FD << "Forward_cpu weight sharing iteration " << i << "\n" ;
              //FD << this->centroids_[this->indices_[i]] << " " ;
              muweight[i] = this->centroids_[this->indices_[i]] ;
          }
      }
      //FD << "\n" ;
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
        if (this->masks_all.size() != 0) {
            //Dtype *weight_diff = this->blobs_[0]->mutable_cpu_diff() ; // added previously!
            int count = this->blobs_[0]->count() ;

            for (int j = 0; j < count; j++) {
                weight_diff[j] *= this->masks_all[j] ; // don't update if mask = 0
            }

            vector<Dtype> tmpDiff(CONV_QUNUM) ;
            vector<int> freq(CONV_QUNUM) ;
            //FD << "@print weight_diff[]\n" ;
            for (int j = 0; j < count; j++) {
                // accumulate here
                if (this->masks_all[j]) {
                    tmpDiff[this->indices_[j]] += weight_diff[j] ;
                    // added by yuzeng
                    // this->centroids_[this->indices_[j]] -= weight_diff[j];
                    //FD << "centroids " << this->centroids_[this->indices_[j]] << "\n" ;
                    freq[this->indices_[j]]++ ;
                }
            }
            //FD << "\n" ;

            for (int j = 0; j < count; j++) {
                // mean (average) of gradient diff???
                if (this->masks_all[j]) {
                    // weight_diff[j] = tmpDiff[this->indices_[j]] / freq[this->indices_[j]] ;
                    // added by yuzeng
                    // MUST use "/freq[]" to update with weight_diff[]'s average!!!
                    this->centroids_[this->indices_[j]] -= LR * weight_diff[j]/freq[this->indices_[j]];
                }
            }
        }

        //for (int j=0; j< CONV_QUNUM; j++){
        //	FD << " centroids: " << this->centroids_[j] <<"\n";
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
