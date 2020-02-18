#ifndef  __MYLAYER_HPP__
#define __MYLAYER_HPP__
#include <iostream>
#include <memory>
#include "myBlob.hpp"

using std::vector;
using std::shared_ptr;

struct Param { // every layer Parameters
	
	// 1. Conv Layer parameters
	int conv_stride;
	int conv_pad;
	int conv_width;
	int conv_height;
	int conv_kernels;
	string conv_weight_init;

	// 2. Pooling Layer parameters
	int pool_stride;
	int pool_width;
	int pool_height;

	// 3. FC Layer parameters
	int fc_kernels;
	string fc_weight_init;

	// 4. Dropout Layer parameters
	double drop_rate;
};
class Layer {
public:
	Layer(){}
	virtual ~Layer() {}
	virtual void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param) = 0;
	virtual void calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) = 0;
	virtual void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode) = 0;
	virtual void backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache, 
						  vector<shared_ptr<Blob>>& grads, const Param& param) = 0;
};

class ConvLayer:public Layer {
public:
	ConvLayer() {}
	~ConvLayer() {}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads, const Param& param);

};

class ReLULayer:public Layer {
public:
	ReLULayer() {}
	~ReLULayer() {}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads, const Param& param);

};

class PoolLayer:public Layer {
public:
	PoolLayer() {}
	~PoolLayer() {}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads, const Param& param);

};

class FCLayer:public Layer {
public:
	FCLayer() {}
	~FCLayer() {}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads, const Param& param);
};

class DropoutLayer :public Layer {
public:
	DropoutLayer() {}
	~DropoutLayer() {}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads, const Param& param);
private:
	shared_ptr<Blob> drop_mask;
};

class SoftmaxLossLayer {
public:
	static void softmax_cross_entropy_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout);
};

class SVMLossLayer {
public:
	static void hinge_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout);
};

#endif  //__MYLAYER_HPP__