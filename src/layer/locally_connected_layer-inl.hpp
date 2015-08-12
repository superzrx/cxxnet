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
          "LocallyConnectedForwardWmatExp: kernel must be smaller than image");
        utils::Check(wshape[srcdim - 1] == ksize_x*ksize_y,
          "LocallyConnectedForwardWmatExp: kernel size does not match weight");
        utils::Check(wshape[srcdim - 3] == ishape[srcdim - 3],
          "LocallyConnectedForwardWmatExp: channel number does not match inputdata");
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


namespace mshadow {
  namespace expr {
    template<typename SrcExp, typename DType, int srcdim>
    struct LocallyConnectedBackpropWmatExp :
      public MakeTensorExp<LocallyConnectedBackpropWmatExp<SrcExp, DType, srcdim>,
      SrcExp, srcdim, DType> {
      const SrcExp &data_in_;
      const SrcExp &data_out_;
      index_t out_shape_y_;
      index_t out_shape_x_;
      index_t in_shape_y_;
      index_t in_shape_x_;
      index_t in_channel_, out_channel_;
      index_t ksize_y_;
      index_t ksize_x_;
      index_t kstride_;
      index_t num_channel_;
      LocallyConnectedBackpropWmatExp(const SrcExp &data_in, const SrcExp &data_out,
        index_t ksize_y, index_t ksize_x, index_t kstride)
        : data_in_(data_in), data_out_(data_out), ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride){
        Shape<srcdim> ishape = ShapeCheck<srcdim, SrcExp>::Check(data_in_);
        Shape<srcdim> oshape = ShapeCheck<srcdim, SrcExp>::Check(data_out_);
        utils::Check(ishape[srcdim - 1] >= ksize_x && ishape[srcdim - 2] >= ksize_y,
          "LocallyConnectedBackpropExp: kernel must be smaller than image");
        utils::Check(oshape[srcdim - 4] == ishape[srcdim - 4],
          "LocallyConnectedBackpropExp: number of samples not match");
        out_shape_y_ = oshape[srcdim - 2];
        out_shape_x_ = oshape[srcdim - 1];
        in_shape_y_ = ishape[srcdim - 2]; 
        in_shape_x_ = ishape[srcdim - 1];
        Shape<srcdim> wshape;
        wshape[srcdim - 4] = oshape[srcdim - 3];
        wshape[srcdim - 3] = ishape[srcdim - 3];
        wshape[srcdim - 2] = oshape[srcdim - 2]*oshape[srcdim-1];
        wshape[srcdim - 1] = ksize_y_*ksize_x_;
        in_channel_ = ishape[srcdim - 3];
        out_channel_ = oshape[srcdim - 3];
        num_channel_ = ishape[srcdim - 4];
        this->shape_ = wshape;
      }
    }; // struct LocallyConnectedBackpropWmatExp

    template<typename SrcExp,typename DType, int etype>
    inline LocallyConnectedBackpropWmatExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      LocallyConnectedBackpropWmat(const Exp<SrcExp, DType, etype> &data_in,
      const Exp<SrcExp, DType, etype> &data_out,
      index_t ksize_y, index_t ksize_x, index_t kstride) {
      return LocallyConnectedBackpropWmatExp<SrcExp,DType, ExpInfo<SrcExp>::kDim>
        (data_in.self(), data_out.self(),
        ksize_y, ksize_x, kstride);
    }

    template<typename SrcExp, typename DType, int srcdim>
    struct Plan<LocallyConnectedBackpropWmatExp<SrcExp,DType, srcdim>, DType> {
    public:
      explicit Plan(const LocallyConnectedBackpropWmatExp<SrcExp, DType, srcdim> &e)
        : data_in_(e.data_in_), data_out_(e.data_out_), num_channel_(e.num_channel_),
        in_shape_y_(e.in_shape_y_), in_shape_x_(e.in_shape_x_), in_channel_(e.in_channel_),
        out_shape_y_(e.out_shape_y_), out_shape_x_(e.out_shape_x_), out_channel_(e.out_channel_),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_){}
      MSHADOW_XINLINE DType Eval(index_t w_i, index_t w_j) const {
        using namespace std;
        // O.C I.C O.H*O.W | K.H*K.W
        //     w_i         |   w_j
        // w_i = (o_c*I.C+i_c)*O.H*O.W + o_y*O.W +o_x
        // w_j = (i_y - i_y_start)*K.W + i_x - i_x_start;
        const index_t o_area = out_shape_y_ * out_shape_x_;
        const index_t o_x = w_i % out_shape_x_;
        const index_t o_y = (w_i % o_area)/out_shape_x_;
        const index_t i_c = (w_i / o_area) % in_channel_;
        const index_t o_c = w_i / (in_channel_ * out_shape_y_ * out_shape_x_);
        const index_t i_y_start = o_y * kstride_;
        const index_t i_x_start = o_x * kstride_;
        const index_t i_y = w_j / ksize_x_ + i_y_start;
        const index_t i_x = w_j % ksize_x_ + i_x_start;
        //input : I.N I.C i.H | I.W 
        //            i_i     | i_j
        // i_i = (o_n*I.C+i_c)*I.H+i_y = o_n*I.C*I.H +i_c*I.H+i_y
        const index_t i_i_1 = in_channel_ * in_shape_y_;
        const index_t i_i_2 = i_c * in_shape_y_ + i_y;
        //output: O.N O.C O.H | O.W
        //            o_i     | o_j
        //i = (o_n*O.C+o_c)*O.H+o_y = o_n*O.C*O.H + o_c*O.H + o_y
        const index_t o_i_1 = out_channel_ * out_shape_y_;
        const index_t o_i_2 = o_c * out_shape_y_ + o_y;
        DType val = static_cast<DType>(0);
        for (index_t i_n = 0; i_n < num_channel_; ++i_n){
          index_t i_i = i_n * i_i_1 + i_i_2;
          index_t o_i = i_n * o_i_1 + o_i_2;
          val += data_out_.Eval(o_i, o_x)*data_in_.Eval(i_i, i_x);
        }
        return val;
      }
    private:
      Plan<SrcExp, DType> data_in_, data_out_;
      const index_t in_channel_, in_shape_y_, in_shape_x_;
      const index_t out_channel_, out_shape_y_, out_shape_x_;
      const index_t ksize_y_, ksize_x_;
      const index_t kstride_, num_channel_;
    }; // struct Plan

  } // namespace expr
} // namespace mshadow

namespace mshadow {
  namespace expr {
    template<typename SrcExp, typename DType, int srcdim>
    struct LocallyConnectedBackpropDataExp :
      public MakeTensorExp<LocallyConnectedBackpropDataExp<SrcExp, DType, srcdim>,
      SrcExp, srcdim, DType> {
      const SrcExp &data_wmat_;
      const SrcExp &data_out_;
      index_t out_shape_y_;
      index_t out_shape_x_;
      index_t out_channel_;
      index_t ksize_y_;
      index_t ksize_x_;
      index_t kstride_;
      LocallyConnectedBackpropDataExp(const SrcExp &data_wmat, const SrcExp &data_out, index_t in_shape_y, index_t in_shape_x,
        index_t ksize_y, index_t ksize_x, index_t kstride)
        : data_wmat_(data_wmat), data_out_(data_out), ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride){
        Shape<srcdim> wshape = ShapeCheck<srcdim, SrcExp>::Check(data_wmat_);
        Shape<srcdim> oshape = ShapeCheck<srcdim, SrcExp>::Check(data_out_);
        utils::Check(in_shape_x >= ksize_x && in_shape_y >= ksize_y,
          "LocallyConnectedBackpropDataExp: kernel must be smaller than image");
        utils::Check(oshape[srcdim - 1] == wshape[srcdim - 4],
          "LocallyConnectedBackpropDataExp: out filter num not consist");
        out_shape_y_ = oshape[srcdim - 2];
        out_shape_x_ = oshape[srcdim - 1];
        out_channel_ = oshape[srcdim - 3];
        Shape<srcdim> ishape;
        ishape[srcdim - 4] = oshape[srcdim - 4];
        ishape[srcdim - 3] = wshape[srcdim - 3];
        ishape[srcdim - 2] = in_shape_y;
        ishape[srcdim - 1] = in_shape_x;
        this->shape_ = ishape;
      }
    }; // struct LocallyConnectedBackpropDataExp

    template<typename SrcExp, typename DType, int etype>
    inline LocallyConnectedBackpropDataExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      LocallyConnectedBackpropData(const Exp<SrcExp, DType, etype> &data_wmat,
      const Exp<SrcExp, DType, etype> &data_out,
      index_t ksize_y, index_t ksize_x, index_t kstride) {
      return LocallyConnectedBackpropDataExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
        (data_wmat.self(), data_out.self(),
        ksize_y, ksize_x, kstride);
    }

    template<typename SrcExp, typename DType, int srcdim>
    struct Plan<LocallyConnectedBackpropDataExp<SrcExp, DType, srcdim>, DType> {
    public:
      explicit Plan(const LocallyConnectedBackpropDataExp<SrcExp, DType, srcdim> &e)
        : wmat_(e.data_wmat_), data_out_(e.data_out_), num_channel_(e.num_channel_),
        in_shape_y_(e.in_shape_y_), in_shape_x_(e.in_shape_x_), in_channel_(e.in_channel_),
        out_shape_y_(e.out_shape_y_), out_shape_x_(e.out_shape_x_), out_channel_(e.out_channel_),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_){}
      MSHADOW_XINLINE DType Eval(index_t i_i, index_t i_j) const {
        using namespace std;
        //input : I.N I.C i.H | I.W 
        //            i_i     | i_j
        // i_i = (i_n*I.C+i_c)*I.H+i_y = o_n*I.C*I.H +i_c*I.H+i_y
        const index_t i_y = i_i % in_shape_y_;
        const index_t i_x = i_j;
        const index_t i_c = (i_i / in_shape_y_) % in_channel_;
        const index_t o_n = i_i / (in_channel_ * in_shape_y_);
        const index_t o_y_start = max(i_y + 1 - ksize_y_, 0) / kstride_;//take i_y as i_y_end
        const index_t o_y_end = i_y / kstride_;//take i_y as i_y_start
        const index_t o_x_start = max(i_x+1-ksize_x_ , 0) / kstride_;//take i_x as i_x_end
        const index_t o_x_end = i_x / kstride_;//take i_x as i_x_start
        
        const index_t o_area = out_shape_y_ *out_shape_x_;
        // O.C I.C O.H*O.W | K.H*K.W
        //     w_i         |   w_j
        // w_i = (o_c*I.C+i_c)*O.H*O.W + o_y*O.W +o_x = o_c*I.C*O.H*O.W +i_c*O.H*O.W +o_y*O.W+o_x
        // w_j = (i_y - i_y_start)*K.W + i_x - i_x_start = 
        const index_t w_i_0 = in_channel_ * o_area;
        const index_t w_i_1 = i_c * o_area;
        //output: O.N O.C O.H | O.W
        //            o_i     | o_j
        //i = (o_n*O.C+o_c)*O.H+o_y = o_n*O.C*O.H + o_c*O.H + o_y
        const index_t o_i_1 = o_n * out_channel_ * out_shape_y_;
        DType val = static_cast<DType>(0);
        for (index_t o_c = 0; o_c < out_channel_; o_c++){
          index_t o_i_2 = o_c * out_shape_y_;
          index_t w_i_2 = o_c * w_i_0;
          for (index_t o_y = o_y_start; o_y >= o_y_end; o_y--){
            index_t o_i = o_i_1 + o_i_2 + o_y;
            index_t w_i_3 = o_y * out_shape_x_;
            for (index_t o_x = o_x_start; o_x >= o_x_end; o_x--){
              index_t w_i = w_i_2 + w_i_1 + w_i_3 + o_x;
              index_t w_j = ( o_y_start - o_y ) * kstride_ * ksize_x_ + (o_x_start - o_x) * kstride_ ;
              val += data_out_.Eval(o_i, o_x) * wmat_.Eval(w_i, w_j);
            }
          }
        }
        return val;
      }
    private:
      Plan<SrcExp, DType> wmat_, data_out_;
      const index_t in_channel_, in_shape_y_, in_shape_x_;
      const index_t out_channel_, out_shape_y_, out_shape_x_;
      const index_t ksize_y_, ksize_x_;
      const index_t kstride_, num_channel_;
    }; // struct Plan

  } // namespace expr
} // namespace mshadow


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

