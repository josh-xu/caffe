#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

// added by xujiang
#ifdef XU_FC
template <typename Dtype>
void InnerProductLayer<Dtype>::ComputeBlobMask(float ratio) {
    // LOG(INFO) is cout???
    std::ofstream FC;
    FC.open("/home/yuzeng/caffe/output_FC.log", std::ios_base::app);
    FC << "FC blob mask" << std::endl ;

    // blobs_[]???
    // count()???
    // CONV_QUNUM???
    //this->msk_no = 3; -- del
    int mask_no = 6;
    #ifdef TWO_BIT
    int mask2b_no = 2;
    #endif
    int count = this->blobs_[0]->count() ;
    this->masks_.resize(count) ;
    this->masks1_.resize(count) ;  
    this->masks2_.resize(count) ;  
    this->masks3_.resize(count) ;
    this->masks4_.resize(count) ;
    this->masks5_.resize(count) ;
    #ifdef TWO_BIT
    this->masks5p6_.resize(count) ;
    this->masks5p7_.resize(count) ;
    #endif
    this->masks_all.resize(count) ;
    this->indices_.resize(count) ;
    this->centroids_.resize(FC_QUNUM) ;

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

    FC << "sort_weight[0]: " << sort_weight[0] << " " <<
                     "sort_weight[count - 1]: " << sort_weight[count - 1] << "\n" ;
    // what's the usage of index???
    int index = int(count * ratio) ; // int(count * (1 - max_weight)) ;
    
    vector<Dtype> thr;
    //thr.resize(mask_no);
    //for (int i = 0; i < this->msk_no; i++){  // set the thr --yuzeng -- del
    FC << "The thrs :" << std::endl;
    for (int i = 0; i < mask_no; i++){  // --del
      thr.push_back(0.2); 
      //FC << thr[i] << "  ";
    }
    //FC << std::endl;
    # ifdef TWO_BIT
    vector<Dtype> thr2b;
    for (int i = 0; i < mask2b_no; i++){  // --del
      thr2b.push_back(0.1); 
    }
    # endif

    // mutable_cpu_data()???
    Dtype *muweight = this->blobs_[0]->mutable_cpu_data() ;
    //float rat = 0 ; // what's the usage of rat???
    vector<float> prune;
    prune.resize(mask_no);
    //float prune[0] = 0;
    //for (int i = 0; i < this->msk_no; i++){  // set the prune number --yuzeng -- del
    for (int i = 0; i < mask_no; i++){  //-- del
      prune.push_back(0); 
    }

    # ifdef TWO_BIT
    vector<float> prune2b;
    for (int i = 0; i < mask2b_no; i++){  //-- del
      prune2b.push_back(0); 
    }
    # endif

    //FC << "masks: " << std::endl;
    if (index > 0) {
        thr[0] = sort_weight[index - 1] ;
        LOG(INFO) << "FC THR: " << thr[0] << " " << ratio << std::endl ;
        for (int i = 0; i < count; i++) {
            // do the masking!!!
            this->masks_[i] = ((weight[i] >= thr[0] || weight[i] < -thr[0])? 1 : 0) ;
            //FC << this->masks_[i];
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
    FC << "percent of 0: " << prune[0] / count << std::endl ;
    FC << "prune[0]: " << prune[0] << std::endl ;
    // min_weight = sort_weight[index] ; // why min_weight is indexed by index???

    //FC << "masks1: " << std::endl;
    for(int i = 0; i < count; i++) {
      float val = 1/pow(2, 1);
      float set_val = (weight[i] > 0)? val : (-1) *val;
      this->masks1_[i] = ((fabs(weight[i]) >= (1-thr[1])*val && fabs(weight[i]) <= (1+thr[1])*val)? 0 : 1) ;
      //FC << this->masks1_[i];
      muweight[i] = ((this->masks1_[i] == 0 && (this->masks1_[i] != 0)) ? set_val :  muweight[i]) ;
      prune[1] += (1 - this->masks1_[i]);
    }

    FC << "percent of 1/2: " << prune[1] / count << std::endl ;
    FC << "prune[1]: " << prune[1] << std::endl ;

    //FC << "masks2: " << std::endl;
    for(int i = 0; i < count; i++) {
      float val = 1/pow(2, 2);
      float set_val = (weight[i] > 0)? val : (-1) *val;
      this->masks2_[i] = ((fabs(weight[i]) >= (1-thr[2])*val && fabs(weight[i]) <= (1+thr[2])*val)? 0 : 1) ;
      //FC << this->masks2_[i];
      muweight[i] = ((this->masks2_[i] == 0 && (this->masks1_[i] != 0)) ? set_val :  muweight[i]) ;
      prune[2] += (1 - this->masks2_[i]);
    }

    FC << "percent of 1/4: " << prune[2] / count << std::endl ;
    FC << "prune[2]: " << prune[2] << std::endl ;

    //FC << "masks3: " << std::endl;
    for(int i = 0; i < count; i++) {
      float val = 1/pow(2, 3);
      float set_val = (weight[i] > 0)? val : (-1) *val;
      this->masks3_[i] = ((fabs(weight[i]) >= (1-thr[3])*val && fabs(weight[i]) <= (1+thr[3])*val)? 0 : 1) ;
      //FC << this->masks3_[i];
      muweight[i] = ((this->masks3_[i] == 0 && (this->masks1_[i] != 0)) ? set_val :  muweight[i]) ;
      prune[3] += (1 - this->masks3_[i]);
    }

    FC << "percent of 1/8: " << prune[3] / count << std::endl ;
    FC << "prune[3]: " << prune[3] << std::endl ;

    //FC << "masks4: " << std::endl;
    for(int i = 0; i < count; i++) {
      float val = 1/pow(2, 4);
      float set_val = (weight[i] > 0)? val : (-1) *val;
      this->masks4_[i] = ((fabs(weight[i]) >= (1-thr[4])*val && fabs(weight[i]) <= (1+thr[4])*val)? 0 : 1) ;
      //FC << this->masks4_[i];
      muweight[i] = ((this->masks4_[i] == 0 && (this->masks1_[i] != 0)) ? set_val :  muweight[i]) ;
      prune[4] += (1 - this->masks4_[i]);
    }

    FC << "percent of 1/16:s " << prune[4] / count << std::endl ;
    FC << "prune[4]: " << prune[4] << std::endl ;
    
    //FC << "masks5: " << std::endl;
    for(int i = 0; i < count; i++) {
      float val = 1/pow(2, 5);
      float set_val = (weight[i] > 0)? val : (-1) *val;
      this->masks5_[i] = ((fabs(weight[i]) >= (1-thr[5])*val && fabs(weight[i]) <= (1+thr[5])*val)? 0 : 1) ;
      //FC << this->masks5_[i];
      muweight[i] = ((this->masks5_[i] == 0 && (this->masks1_[i] != 0)) ? set_val :  muweight[i]) ;
      prune[5] += (1 - this->masks5_[i]);
    }

    FC << "percent of 1/32:s " << prune[5] / count << std::endl ;
    FC << "prune[5]: " << prune[5] << std::endl ;

    # ifdef TWO_BIT
    //FC << "masks5p6: " << std::endl;
    for(int i = 0; i < count; i++) {
      float val = 1/pow(2, 5) + 1/pow(2, 6);
      float set_val = (weight[i] > 0)? val : (-1) *val;
      this->masks5p6_[i] = ((fabs(weight[i]) >= (1-thr2b[0])*val && fabs(weight[i]) <= (1+thr2b[0])*val)? 0 : 1) ;
      //FC << this->masks5_[i];
      muweight[i] = ((this->masks5p6_[i] == 0 && (this->masks1_[i] != 0)) ? set_val :  muweight[i]) ;
      prune2b[0] += (1 - this->masks5p6_[i]);
    }

    FC << "percent of 1/32+1/64:s " << prune2b[0] / count << std::endl ;
    FC << "prune2b[0]: " << prune2b[0] << std::endl ;

    for(int i = 0; i < count; i++) {
      float val = 1/pow(2, 5) + 1/pow(2, 7);
      float set_val = (weight[i] > 0)? val : (-1) *val;
      this->masks5p7_[i] = ((fabs(weight[i]) >= (1-thr2b[1])*val && fabs(weight[i]) <= (1+thr2b[1])*val)? 0 : 1) ;
      //FC << this->masks5_[i];
      muweight[i] = ((this->masks5p7_[i] == 0 && (this->masks1_[i] != 0)) ? set_val :  muweight[i]) ;
      prune2b[1] += (1 - this->masks5p7_[i]);
    }

    FC << "percent of 1/32+1/128:s " << prune2b[1] / count << std::endl ;
    FC << "prune2b[1]: " << prune2b[1] << std::endl ;
    # endif

    // initialize the masks_all;

    for (int i = 0; i < count; i++) {
      this->masks_all.push_back(1);
    }
    
    //FC << "masks_all :" <<std::endl;
    for (int i = 0; i < count; i++){
      //for(int j = 0; j < mask_no; j++)
      this->masks_all[i] = this->masks_[i] & this->masks1_[i] & this->masks2_[i] & this->masks3_[i] & this->masks4_[i] & this->masks5_[i];
      //FC << this->masks_all[i];
      # ifdef TWO_BIT
      this->masks_all[i] *= this->masks5p6_[i];
      this->masks_all[i] *= this->masks5p7_[i];
      # endif
    }

    // kmeans_cluster()???
    int nCentroid = FC_QUNUM ;
    if (nCentroid > count) {
        //FC << "@@@ Weird Things Happened!!!\n" ;
        assert(false && "nCentroid > count") ;
        nCentroid = count ;
    }
    FC << "nCentroid = FC_QUNUM: " << nCentroid << "\n" ;
    FC << "nWeights = count: " << count << "\n" ;
    kmeans_cluster(this->indices_, this->centroids_, muweight, count,
                       this->masks_all, nCentroid, 1000) ;
    // added by yuzeng
    float sparsity_post = 0;
    for (int i = 0; i < count; i++ ){
      //std::cout << this->indices_[i];
      if (this->indices_[i] == -1)
        sparsity_post += 1;
    }
    FC << "sparsity after kmeans " << sparsity_post / count << std::endl;
    FC << "prune all: " << sparsity_post << std::endl ;
    FC << "################# The end of FC layer data ####################" << std::endl;
    FC.close();
}
#endif

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // added by xujiang
  #ifdef XU_FC
  if (this->masks_all.size() != 0) {
      Dtype *muweight = this->blobs_[0]->mutable_cpu_data() ;
      int count = this->blobs_[0]->count() ;

      const Dtype *weight = this->blobs_[0]->cpu_data() ;
      /*
      std::ofstream CL;
      CL.open("/home/yuzeng/caffe/FC_clusters.log", std::ios_base::app);
      CL << "Clusters are :" << std::endl ;
      for (int i=0; i < CONV_QUNUM; i++){
          CL << "Centroid: " << this->centroids_[i] << std::endl ;
          std::cout << "weight with centroids" << std::endl;
          for (int j = 0; j < count; j++ ){
              if (this->masks_all[j]) {std::cout << weight[i] << "\t";}
              if (this->indices_[j] == i){
                  CL << weight[i];
              }
          }
          CL << "##################" << std::endl << std::endl ;
      }
      CL.close();
      */

      for (int i = 0; i < count; i++) {
          if (this->masks_all[i]) {
              // weight sharing!!!
              muweight[i] = this->centroids_[this->indices_[i]] ;
          }
      }
  }
  #endif
  // added by xujiang

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  // added by xujiang
  #ifdef XU_FC
  int count = this->blobs_[0]->count() ;
  Dtype *weight_diff = this->blobs_[0]->mutable_cpu_diff() ;
  #endif
  // added by xujiang

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }

    // added by xujiang
    #ifdef XU_FC
    if (this->masks_all.size() != 0) {
        for (int j = 0; j < count; j++) {
            weight_diff[j] *= this->masks_all[j] ; // don't update if mask = 0
        }
    } else {
        // supress warning by compiler
        if (count) {} ;
        if (weight_diff[0]) {} ;
    }
    #endif
    // added by xujiang
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();

    // added by xujiang
    #ifdef XU_FC
    if (this->masks_all.size() != 0) {
        vector<Dtype> tmpDiff(FC_QUNUM) ;
        vector<int> freq(FC_QUNUM) ;
        for (int j = 0; j < count; j++) {
            // accumulate here
            if (this->masks_all[j]) {
                tmpDiff[this->indices_[j]] += weight_diff[j] ;
                // added by yuzeng
                //this->centroids_[this->indices_[j]] -= weight_diff[j];
                freq[this->indices_[j]]++ ;
            }
        }
        for (int j = 0; j < count; j++) {
            // mean (average) of gradient diff???
            if (this->masks_all[j]) {
                //weight_diff[j] = tmpDiff[this->indices_[j]] / freq[this->indices_[j]] ;
                //added by yuzeng
                this->centroids_[this->indices_[j]] -= LR * weight_diff[j]/freq[this->indices_[j]];
            }
        }
    } else {
        // supress warning by compiler
        if (count) {} ;
        if (weight_diff[0]) {} ;
    }
    #endif
    // added by xujiang

    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
