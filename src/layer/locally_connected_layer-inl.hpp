/*!
*  Copyright (c) 2014 by Contributors
* \author Ruixin Zhang(ruixinzhang@tencent.com), Hangyu Yan(hangyuyan@tencent.com)
*/

#ifndef CXXNET_LAYER_LOCAL_CONVOLUTION_LAYER_INL_HPP_
#define CXXNET_LAYER_LOCAL_CONVOLUTION_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include <string>
namespace mshadow {
  namespace expr {

    template<typename SrcExp,typename DType, int srcdim>
    struct LocallyConnectedForwardExp :
      public MakeTensorExp<LocallyConnectedForwardExp<SrcExp,DType,srcdim>,
      SrcExp, srcdim, DType> {
      const SrcExp &data_in_;
      const SrcExp &wmat_;
      index_t ksize_y_;
      index_t ksize_x_;
      index_t kstride_;
      index_t in_height_;
      index_t in_width_;
      index_t in_channels_;
      LocallyConnectedForwardExp(const SrcExp &data_in,const SrcExp& wmat,
        index_t ksize_y, index_t ksize_x, index_t kstride)
        : data_in_(data_in), wmat_(wmat),
        ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride) {
        Shape<srcdim> ishape = ShapeCheck<srcdim, SrcExp>::Check(data_in_);
        Shape<srcdim> wshape = ShapeCheck<srcdim, SrcExp>::Check(wmat_);
        utils::Check(ishape[srcdim - 1] >= ksize_x && ishape[srcdim - 2] >= ksize_y,
          "LocallyConnectedForwardExp: kernel must be smaller than image");
        utils::Check(wshape[srcdim - 1] == ksize_x*ksize_y,
          "LocallyConnectedForwardExp: kernel size does not match weight");
        utils::Check(wshape[srcdim - 3] == ishape[srcdim - 3],
          "LocallyConnectedForwardExp: channel number does not match inputdata");
        this->in_height_ = ishape[srcdim - 2];
        this->in_width_ = ishape[srcdim - 1];
        this->in_channels_ = ishape[srcdim - 3];
        this->shape_[srcdim - 4] = wshape[srcdim - 3];
        this->shape_[srcdim - 3] = ishape[srcdim - 3];
        this->shape_[srcdim - 2] = (in_height_ - ksize_y) / kstride + 1;
        this->shape_[srcdim - 1] = (in_width_ - ksize_x) / kstride + 1;
      }
    }; // struct LocallyConnectedForwardExp
    /*!
    * \brief multiply data_in and wmat within a subregion and sum together
    * \param src source image, shape: (batch, channel, height, width)
    * \param ksize_y kernel size in height
    * \param ksize_x kernel size in width
    * \param kstride stride for each kernel
    * \return expression of LocallyConnectedForward result
    * \tparam SrcExp source expression
    * \tparam DType the content data type
    * \tparam etype type of expression
    */
    template<typename SrcExp,typename DType, int etype>
    inline LocallyConnectedForwardExp< SrcExp, DType, ExpInfo<SrcExp>::kDim>
      LocallyConnectedForward(const Exp<SrcExp, DType, etype> &data_in, const Exp<SrcExp, DType, etype> &wmat,
      index_t ksize_y, index_t ksize_x, index_t kstride) {
      TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>::Error_Expression_Does_Not_Meet_Dimension_Req();
      return LocallyConnectedForwardExp<SrcExp,DType, ExpInfo<SrcExp>::kDim>
        (data_in.self(), wmat.self(), ksize_y, ksize_x, kstride);
    }


    template<typename SrcExp,typename DType, int srcdim>
    struct Plan<LocallyConnectedForwardExp<SrcExp,  DType, srcdim>, DType> {
    public:
      explicit Plan(const LocallyConnectedForwardExp<SrcExp,DType, srcdim> &e)
        : data_in_(MakePlan(e.data_in_)),
        wmat_(MakePlan(e.wmat_)),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_),
        in_height_(e.in_height_), in_width_(e.in_width_), in_channel_(e.in_channels_),
        out_height_(e.shape_[srcdim - 2]), out_width_(e.shape_[srcdim - 1]), out_channel_(e.shape_[srcdim - 3]) {}

      MSHADOW_XINLINE DType Eval(index_t o_i, index_t o_j) const {
        using namespace std;
        //output: O.N O.C O.H | O.W
        //            o_i     | o_j
        //i = (o_n*O.C+o_c)*O.H+o_y = o_n*O.C*O.H + o_c*O.H + o_y

        const index_t o_n = o_i / (out_channel_*out_height_);
        const index_t o_c = o_i / (out_height_)-o_n*out_channel_;
        const index_t o_y = o_i - o_n*out_channel_*out_height_ - o_c*out_height_;
        const index_t o_x = o_j;

        //weight: O.C I.C O.H*O.W | K.H*K.W
        //            w_i         |   w_j
        // w_i = (o_c*I.C+i_c)*O.H*O.W+o_y*O.W+o_x = o_c*I.C*O.H*O.W+o_y*O.W+o_x  +  i_c*O.H*O.W
        // w_j = (i_y-i_y_start)*K.W +(i_x-i_x_start)
        const index_t o_area = out_height_*out_width_;
        const index_t weight_offset = o_c*in_channel_*o_area + o_y*out_width_ + o_x;

        //input : I.N I.C i.H | I.W 
        //            i_i     | i_j
        // i_i = (o_n*I.C+i_c)*I.H+i_y = o_n*I.C*I.H +i_c*I.H+i_y
        const index_t in_offset = o_n*in_channel_*in_height_;
        const index_t i_y_start = o_y * kstride_;
        const index_t i_y_end = min(i_y_start + ksize_y_, in_height_);
        const index_t i_x_start = o_x * kstride_;
        const index_t i_x_end = min(i_x_start + ksize_x_, in_width_);

        DType res=static_cast<DType>(0);
        for (index_t i_y = i_y_start; i_y < i_y_end; ++i_y) {
          for (index_t i_x = i_x_start; i_x < i_x_end; ++i_x) {
            for (index_t i_c = 0; i_c < in_channel_; i_c++){
              index_t i_i = in_offset+i_c*in_height_+i_y;
              index_t w_i = weight_offset + i_c*o_area;
              index_t w_j = (i_y-i_y_start)*ksize_x_+(i_x-i_x_start);
              res+=data_in_.Eval(i_i,i_x)*wmat_.Eval(w_i,w_j);
            }
          }
        }
        return res;
      }
    private:
      Plan<SrcExp, DType> data_in_,wmat_;
      const index_t ksize_y_, ksize_x_, kstride_;
      const index_t in_channel_,in_height_, in_width_;
      const index_t out_channel_,out_height_, out_width_;
    }; // struct Plan

  } // namespace expr
} // namespace mshadow

/*
namespace mshadow {
  namespace expr {

    template<typename Reducer, typename SrcExp, typename MaskExp, typename DType, int srcdim>
    struct InsanityUnPoolingExp :
      public MakeTensorExp<InsanityUnPoolingExp<Reducer, SrcExp, MaskExp, DType, srcdim>,
      SrcExp, srcdim, DType> {
      const SrcExp &data_src_;
      const SrcExp &data_pooled_;
      const SrcExp &grad_pooled_;
      const MaskExp &mask_;
      index_t pshape_y_;
      index_t pshape_x_;
      index_t ksize_y_;
      index_t ksize_x_;
      index_t kstride_;
      DType p_keep_;
      InsanityUnPoolingExp(const SrcExp &data_src,
        const SrcExp &data_pooled,
        const SrcExp &grad_pooled,
        const MaskExp &mask,
        index_t ksize_y, index_t ksize_x, index_t kstride, DType p_keep)
        : data_src_(data_src), data_pooled_(data_pooled), grad_pooled_(grad_pooled),
        mask_(mask), ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride), p_keep_(p_keep) {
        Shape<srcdim> pshape = ShapeCheck<srcdim, SrcExp>::Check(grad_pooled);
        utils::Check(pshape == ShapeCheck<srcdim, SrcExp>::Check(data_pooled),
          "UnPoolingExp: pooled shape mismatch");
        Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(data_src);
        Shape<srcdim> smshape = ShapeCheck<srcdim, MaskExp>::Check(mask_);
        utils::Check(sshape == smshape, "Incorrect shape");
        for (int k = 0; k < srcdim - 2; ++k) {
          utils::Check(pshape[k] == sshape[k],
            "UnPoolingExp: pool and src shape mismatch");
        }
        pshape_x_ = pshape[srcdim - 1];
        pshape_y_ = pshape[srcdim - 2];
        this->shape_ = sshape;
      }
    }; // struct InsanityUnPoolingExp

    template<typename Reducer, typename SrcExp, typename MaskExp, typename DType, int etype>
    inline InsanityUnPoolingExp<Reducer, SrcExp, MaskExp, DType, ExpInfo<SrcExp>::kDim>
      insanity_unpool(const Exp<SrcExp, DType, etype> &data_src,
      const Exp<SrcExp, DType, etype> &data_pooled,
      const Exp<SrcExp, DType, etype> &grad_pooled,
      const Exp<MaskExp, DType, etype> &mask,
      index_t ksize_y, index_t ksize_x, index_t kstride, DType p_keep) {
      return InsanityUnPoolingExp<Reducer, SrcExp, MaskExp, DType, ExpInfo<SrcExp>::kDim>
        (data_src.self(), data_pooled.self(), grad_pooled.self(), mask.self(),
        ksize_y, ksize_x, kstride, p_keep);
    }

    template<typename Reducer, typename SrcExp, typename MaskExp, typename DType, int srcdim>
    struct Plan<InsanityUnPoolingExp<Reducer, SrcExp, MaskExp, DType, srcdim>, DType> {
    public:
      explicit Plan(const InsanityUnPoolingExp<Reducer, SrcExp, MaskExp, DType, srcdim> &e)
        : data_src_(e.data_src_), data_pooled_(e.data_pooled_), grad_pooled_(e.grad_pooled_),
        mask_(e.mask_), sshape_y_(e.shape_[srcdim - 2]), sshape_x_(e.shape_[srcdim - 3]),
        pshape_y_(e.pshape_y_), pshape_x_(e.pshape_x_),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_),
        p_keep_(e.p_keep_), delta_((1.0f - p_keep_) / 4.0f) {}
      MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
        using namespace std;
        const index_t x = j;
        const index_t y = i % sshape_y_;
        const index_t c = i / sshape_y_;
        const DType flag = mask_.Eval(i, j);
        index_t loc_x = x;
        index_t loc_y = y;
        if (flag < p_keep_) {
          ;
        }
        else if (flag < p_keep_ + delta_) {
          loc_y = loc_y > 0 ? loc_y - 1 : loc_y;
        }
        else if (flag < p_keep_ + delta_ * 2.0f) {
          loc_y = loc_y + 1 < sshape_y_ ? loc_y + 1 : sshape_y_ - 1;
        }
        else if (flag < p_keep_ + delta_ * 3.0f) {
          loc_x = loc_x > 0 ? loc_x - 1 : loc_x;
        }
        else {
          loc_x = loc_x + 1 < sshape_x_ ? loc_x + 1 : sshape_x_ - 1;
        }
        const DType vsrc = data_src_.Eval(c * sshape_y_ + loc_y, loc_x);
        const index_t py_min =
          y < ksize_y_ ? 0 : (y - ksize_y_ + kstride_) / kstride_;
        const index_t px_min =
          x < ksize_x_ ? 0 : (x - ksize_x_ + kstride_) / kstride_;
        const index_t py_max = min((y + kstride_) / kstride_, pshape_y_);
        const index_t px_max = min((x + kstride_) / kstride_, pshape_x_);
        DType val = static_cast<DType>(0);
        for (index_t py = py_min; py < py_max; ++py) {
          for (index_t px = px_min; px < px_max; ++px) {
            val += Reducer::PartialGrad(vsrc,
              data_pooled_.Eval(c * pshape_y_ + py, px)) *
              grad_pooled_.Eval(c * pshape_y_ + py, px);
          }
        }
        return val;
      }
    private:
      Plan<SrcExp, DType> data_src_, data_pooled_, grad_pooled_;
      Plan<MaskExp, DType> mask_;
      const index_t sshape_y_, sshape_x_, pshape_y_, pshape_x_;
      const index_t ksize_y_, ksize_x_;
      const index_t kstride_;
      const DType p_keep_;
      const DType delta_;
    }; // struct Plan

  } // namespace expr
} // namespace mshadow
*/
namespace cxxnet {
namespace layer {

template<typename xpu>
class LocallyConnectedLayer : public ILayer<xpu> {
 public:
  LocallyConnectedLayer(mshadow::Random<xpu> *p_rnd)
	  : prnd_(p_rnd), wmat_(false), gwmat_(false) {}

  virtual ~LocallyConnectedLayer(void) {}

  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam(name, val);
  }

  virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
	  pvisitor->Visit("wmat", wmat_, gwmat_);
  }

  virtual void InitModel(void) {
	  wmat_.Resize(wmat_shape_);
	  param_.RandInitWeight(this->prnd_, wmat_, wmat_shape_[1], wmat_shape_[0]);
	  // setup gradient
	  gwmat_.Resize(wmat_shape_);
	  gwmat_ = 0.0f; 
  }

  virtual void SaveModel(utils::IStream &fo) const {
	  fo.Write(&param_, sizeof(LayerParam));
	  wmat_.SaveBinary(fo);
  }

  virtual void LoadModel(utils::IStream &fi) {
	  utils::Check(fi.Read(&param_, sizeof(LayerParam)) != 0,
		  "LocallyConnectedLayer: LoadModel invalid model file");
	  wmat_.LoadBinary(fi);
	  // setup gradient
	  gwmat_.Resize(wmat_.shape_);
	  gwmat_ = 0.0f; 
  }

  virtual void SetStream(mshadow::Stream<xpu> *stream) {
	  // stream of wmat and bias may be reset, but it is ok
	  wmat_.set_stream(stream);
	  gwmat_.set_stream(stream);
  }

  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    InitNode(nodes_in, nodes_out, p_cstate);
  }

  virtual void OnBatchSizeChanged(const std::vector<Node<xpu>*> &nodes_in,
                                  const std::vector<Node<xpu>*> &nodes_out,
                                  ConnectState<xpu> *p_cstate) {
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
	  const int ksize_y = param_.kernel_height;
	  const int ksize_x = param_.kernel_width;
	  const int pad_y = param_.pad_y;
	  const int pad_x = param_.pad_x;
	  nodes_out[0]->data = LocallyConnectedForward(pad(nodes_in[0]->data, pad_y, pad_x), pad(wmat_, 0, 0), ksize_y, ksize_x, param_.stride);
    if (param_.no_bias == 0){
      for (size_t i = 0; i < nodes_out[0]->data.shape_[0]; i++){
        nodes_out[0]->data.Slice(i, i + 1) += reshape(bias_, mshadow::Shape4(1, bias_shape_[0], bias_shape_[1], bias_shape_[2]));
      }
    }
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    
      const int ksize_y = param_.kernel_height;
      const int ksize_x = param_.kernel_width;
      const int pad_y = param_.pad_y;
      const int pad_x = param_.pad_x;
	  mshadow::Shape<2> pshape = nodes_out[0]->data[0][0].shape_;
	  mshadow::Shape<4> sshape = nodes_in[0]->data.shape_;
	  sshape[2] += param_.pad_y * 2;
	  sshape[3] += param_.pad_x * 2;
    //gwmat_;
    //gwmat_ += LocallyConnectedLayerBackwardGwmat(pad(nodes_in[0]->data, pad_y, pad_x));
    if (prop_grad) {
      //nodes_in[0]->data = LocalConvolutionBackwardData(pad(nodes_in[0]->data, pad_y, pad_x), pshape, pad(wmat_, 0, 0), ksize_y, ksize_x, param_.stride);
    }
  }

 protected:
  inline void InitNode(const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "LocallyConnectedLayer: only support 1-1 connection");

	const index_t ksize_y = static_cast<index_t>(param_.kernel_height);
	const index_t ksize_x = static_cast<index_t>(param_.kernel_width);
	const index_t kstride = static_cast<index_t>(param_.stride);
	mshadow::Shape<4> ishape = nodes_in[0]->data.shape_;
	
	utils::Check(param_.num_channel > 0, "LocallyConnectedLayer: must set nchannel correctly");
	utils::Check(param_.kernel_height > 0 && param_.kernel_width > 0, "LocallyConnectedLayer: must set kernel_size correctly");
	utils::Check(ksize_x <= ishape[3] && ksize_y <= ishape[2], "LocallyConnectedLayer: kernel size exceed input");
	if (param_.num_input_channel == 0) {
		param_.num_input_channel = static_cast<int>(ishape[1]);
	}
	else {
		utils::Check(param_.num_input_channel == static_cast<int>(ishape[1]),
			"LocallyConnectedLayer: number of input channels is not consistent");
	}

	mshadow::Shape<4> oshape = mshadow::
		Shape4(ishape[0], param_.num_channel,
		(ishape[2] + 2 * param_.pad_y - ksize_y) / kstride + 1,
		(ishape[3] + 2 * param_.pad_x - ksize_x) / kstride + 1);
	nodes_out[0]->data.shape_ = oshape;

	//Init in_shape_
	in_shape_[0] = ishape[2];
	in_shape_[1] = ishape[3];
	//Init wmat_shape_
	wmat_shape_[0] = oshape[1];
	wmat_shape_[1] = ishape[1];
	wmat_shape_[2] = oshape[2] * oshape[3];
	wmat_shape_[3] = ksize_x * ksize_y;

  bias_shape_[0] = oshape[1];
  bias_shape_[1] = oshape[2];
  bias_shape_[2] = oshape[3];
	// use 3 temp state 
	//p_cstate->states.resize(3);
	
  }

  /*! \brief random number generator */
  mshadow::Random<xpu> *prnd_;
  /*! \brief parameters that potentially be useful */
  LayerParam param_;
  mshadow::Shape<2> in_shape_;
  /*! \brief weight matrix */
  mshadow::TensorContainer<xpu, 4> wmat_;
  /*! \brief accumulates the gradient of weight matrix */
  mshadow::TensorContainer<xpu, 4> gwmat_; 

  mshadow::TensorContainer<xpu, 3> bias_;
  /*! \brief accumulates the gradient of weight matrix */
  mshadow::TensorContainer<xpu, 3> gbias_;
  //shape for w
  mshadow::Shape<4> wmat_shape_;
  mshadow::Shape<3> bias_shape_;
}; // class LocallyConnectedLayer
}  // namespace layer
}  // namespace cxxnet
#endif  // CXXNET_LAYER_LOCAL_CONVOLUTION_LAYER_INL_HPP_

