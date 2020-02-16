#include "myLayer.hpp"
#include <cassert>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace arma;


// 将arma::mat转成cv::mat
template<typename T>
void Arma_mat2cv_mat(const arma::Mat<T>& arma_mat_in, cv::Mat_<T>& cv_mat_out) {
	cv::transpose(cv::Mat_<T>(
		static_cast<int>(arma_mat_in.n_cols),
		static_cast<int>(arma_mat_in.n_rows),
		const_cast<T*>(arma_mat_in.memptr())
		), cv_mat_out);
	return;
};

void visiable(const cube& in, vector<cv::Mat_<double>>& vec_mat) {
	int num = in.n_slices;
	for (int i = 0; i < num; i++) {
		cv::Mat_<double> mat_cv;
		arma::mat mat_arma = in.slice(i);
		Arma_mat2cv_mat<double>(mat_arma, mat_cv);;
		vec_mat.push_back(mat_cv);
	}
	return;
}

///////////////////////////////////initLayer/////////////////////////////////////////
void ConvLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param) {
	// 1. 获取卷积核尺寸(F, C, H, W)
	int tF = param.conv_kernels;
	int tC = inShape[1];
	int tH = param.conv_height;
	int tW = param.conv_width;
	
	// 2. 初始化存储weight和bias的Blob (in[1], in[2]) = (w, b)
	if (!in[1]) { // 存储weight的blob不为空
		in[1].reset(new Blob(tF, tC, tH, tW, TRANDN));
		(*in[1]) *= 1e-2;
	}
	if (!in[2]) { // 存储bias的blob不为空
		in[2].reset(new Blob(tF, 1, 1, 1, TRANDN));
		(*in[2]) *= 1e-2;
		
	}

	return;
}

void ReLULayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param) {
	return;
}

void PoolLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param) {

	return;
}

void FCLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param) {
		// 1. 获取全连接核的尺寸(F, C, H, W)
		int tF = param.fc_kernels;
		int tC = inShape[1];
		int tH = inShape[2];
		int tW = inShape[3];

		// 2. 初始化存储weight和bias的Blob (in[1], in[2]) = (w, b)
		if (!in[1]) { // 存储weight的blob不为空
			in[1].reset(new Blob(tF, tC, tH, tW, TRANDN));
			(*in[1]) *= 1e-2;
			
		}
		if (!in[2])// 存储bias的blob不为空
			in[2].reset(new Blob(tF, 1, 1, 1, TZEROS));
		return;
}
///////////////////////////////////calcShape/////////////////////////////////////////
void ConvLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	// 1. 获取输入Blob尺寸
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];
	
	// 2. 获取卷积核尺寸
	int tF = param.conv_kernels; // kernel numers
	int tH = param.conv_height;  // kernel height
	int tW = param.conv_width;   // kernel width
	int tP = param.conv_pad;     // kernel padding
	int tS = param.conv_stride;  // kernel stride

	// 3. calc Conved shape
	int No = Ni;
	int Co = tF;
	int Ho = (Hi + (tP << 1) - tH) / tS + 1;
	int Wo = (Wi + (tP << 1) - tW) / tS + 1;

	// 4. 赋值输出Blob尺寸
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}

void ReLULayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	outShape.assign(inShape.begin(), inShape.end()); // 将inShape复制一份给outShape（深拷贝）
	return;
}

void PoolLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	// 1. 获取输入Blob尺寸
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];

	// 2. 获取卷积核尺寸
	int tH = param.pool_height;  // kernel height
	int tW = param.pool_width;   // kernel width
	int tS = param.pool_stride;  // kernel stride

	// 3. calc Pooled Shape
	int No = Ni;
	int Co = Ci;
	int Ho = (Hi - tH) / tS + 1;
	int Wo = (Wi - tW) / tS + 1;

	// 4. 赋值输出Blob尺寸
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}
void FCLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	// 1. 获取输入Blob的尺寸
	int No = inShape[0]; // batch size
	int Co = param.fc_kernels; // current layer nn numbers
	int Ho = 1;
	int Wo = 1;

	// 2. 赋值
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}
///////////////////////////////////forward///////////////////////////////////
void ConvLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param) {
	if (out)
		out.reset();
	// 1. 获取相关参数（输入，卷积核，输出）
	assert(in[0]->getC() == in[1]->getC());
	int N = in[0]->getN();   // input Blob中cube的个数
	int C = in[0]->getC();   // input Blob中的channel个数
	int Hx = in[0]->getH();  // input Blob的高
	int Wx = in[0]->getW();  // input Blob的宽

	int F = in[1]->getN();    // number of kernels
	int Hw = in[1]->getH();   // kernel's height
	int Ww = in[1]->getW();   // kernel's width

	int Ho = (Hx + (param.conv_pad << 1) - Hw) / param.conv_stride + 1; // conved blob height
	int Wo = (Wx + (param.conv_pad << 1) - Ww) / param.conv_stride + 1; // conved Blob width

	// 2. Padding
	Blob padX = in[0]->pad(param.conv_pad);


	// 3. Convlution
	out.reset(new Blob(N, F, Ho, Wo));
	for (int n = 0; n < N; n++) {
		for (int f = 0; f < F; f++) {
			for (int hh = 0; hh < Ho; hh++) {
				for (int ww = 0; ww < Wo; ww++) {
					cube window = padX[n](span(hh * param.conv_stride, hh * param.conv_stride + Hw - 1),
										  span(ww * param.conv_stride, ww * param.conv_stride + Ww - 1),
						                  span::all);
					// out = wx+b
					(*out)[n](hh, ww, f) = accu(window % (*in[1])[f]) + as_scalar((*in[2])[f]);
				}
			}
		}
	}
	//vector<cv::Mat_<double>> vec_mat_out;
	//visiable((*out)[0], vec_mat_out); // 可视化第一个卷积核
 	return;
}

void ReLULayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param) {
	if (out)
		out.reset();
	out.reset(new Blob(*in[0]));
	out->maxIn(0);
	return;
}

void PoolLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param) {
	if (out)
		out.reset();
	// 1. 获取相关参数（输入，池化核，输出）
	int N = in[0]->getN();   // input Blob中cube的个数
	int C = in[0]->getC();   // input Blob中的channel个数
	int Hx = in[0]->getH();  // input Blob的高
	int Wx = in[0]->getW();  // input Blob的宽

	int Hw = param.pool_height;   // kernel's height
	int Ww = param.pool_width;   // kernel's width

	int Ho = (Hx - Hw) / param.pool_stride + 1; // Pooled blob height
	int Wo = (Wx - Ww) / param.pool_stride + 1; // Pooled Blob width

	// 2. Pool
	out.reset(new Blob(N, C, Ho, Wo));
	for (int n = 0; n < N; n++)
		for (int c = 0; c < C; c++)
			for (int hh = 0; hh < Ho; hh++)
				for (int ww = 0; ww < Wo; ww++)
					(*out)[n](hh, ww, c) = (*in[0])[n](span(hh * param.pool_stride, hh * param.pool_stride + Hw - 1),
						span(ww * param.pool_stride, ww * param.pool_stride + Ww - 1),
						span(c, c)).max();
	return;
}

void FCLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param) {



	if (out)
		out.reset();
	// 1. 获取相关参数（输入，全连接核，输出）

	int N = in[0]->getN();   // input Blob中cube的个数
	int C = in[0]->getC();   // input Blob中的channel个数
	int Hx = in[0]->getH();  // input Blob的高
	int Wx = in[0]->getW();  // input Blob的宽

	int F = in[1]->getN();    // number of kernels
	int Hw = in[1]->getH();   // kernel's height
	int Ww = in[1]->getW();   // kernel's width
	assert(in[0]->getC() == in[1]->getC());
	assert(Hx == Hw && Wx == Ww);

	int Ho = 1;
	int Wo = 1;

	// 3. FC
	out.reset(new Blob(N, F, Ho, Wo));

	for (int n = 0; n < N; n++) 
		for (int f = 0; f < F; f++) 
			(*out)[n](0, 0, f) = accu((*in[0])[n] % (*in[1])[f]) + as_scalar((*in[2])[f]);
	return;
}

void SoftmaxLossLayer::softmax_cross_entropy_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout) {

	if (dout)
		dout.reset();

	// 1. 获取相关尺寸
	int N = in[0]->getN();
	int C = in[0]->getC();
	int Hx = in[0]->getH();
	int Wx = in[0]->getW();
	assert(Hx == 1 && Wx == 1);
	
	dout.reset(new Blob(N, C, Hx, Wx)); // (N, C, 1, 1)
	double loss_ = 0;
	for (int i = 0; i < N; i++) {
		// softmax
		cube prob = arma::exp((*in[0])[i]) / arma::accu(arma::exp((*in[0])[i]));
		loss_ += (-arma::accu((*in[1])[i] % arma::log(prob)));
		// 梯度表达式推导
		(*dout)[i] = prob - (*in[1])[i]; // 计算各个样本产生的误差信号（反向梯度）

	}
	loss = loss_ / N;
	return;
}

///////////////////////////////////backward///////////////////////////////////
void FCLayer::backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads, const Param& param) {
	

	// dx, dw, db
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));
	grads[1].reset(new Blob(cache[1]->size(), TZEROS));
	grads[2].reset(new Blob(cache[2]->size(), TZEROS));

	int N = grads[0]->getN();
	int F = grads[1]->getN();
	assert(F == cache[1]->getN());

	for (int n = 0; n < N; n++) {
		for (int f = 0; f < F; f++) {
			// dx
			(*grads[0])[n] += (*din)[n](0, 0, f) * (*cache[1])[f];
			// dw
			(*grads[1])[f] += (*din)[n](0, 0, f)  * (*cache[0])[n] / N;
			// db
			(*grads[2])[f] += (*din)[n](0, 0, f) / N;
		}
	}
	return;
}

void PoolLayer::backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads, const Param& param) {

	// 1. 设置输出梯度Blob的尺寸 (dx----grdas[0])
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));
	// 2. 获取输入梯度Blob的尺寸
	int Nd = din->getN();        //输入梯度Blob中cube个数（该batch样本个数）
	int Cd = din->getC();         //输入梯度Blob通道数
	int Hd = din->getH();      //输入梯度Blob高
	int Wd = din->getW();    //输入梯度Blob宽
	// 3. 获取池化核相关参数
	int Hp = param.pool_height;
	int Wp = param.pool_width;
	int stride = param.pool_stride;

	// 4. 开始反向传播
	for (int n = 0; n < Nd; n++) { // 输出cube数
		for (int c = 0; c < Cd; c++) { // 输出通道数
			for (int hh = 0; hh < Hd; hh++) { // 输出Blob的高
				for (int ww = 0; ww < Wd; ww++) { // 输出Blob的宽
					// (1)获取掩码mask
					mat window = (*cache[0])[n](span(hh * param.pool_stride, hh * param.pool_stride + Hp - 1),
						span(ww * param.pool_stride, ww * param.pool_stride + Wp - 1),
						span(c, c));
					double maxv = window.max();
					mat mask = conv_to<mat>::from(maxv == window); // umat -> mat

					(*grads[0])[n](span(hh * param.pool_stride, hh * param.pool_stride + Hp - 1),
						span(ww * param.pool_stride, ww * param.pool_stride + Wp - 1),
						span(c, c)) += mask * (*din)[n](hh, ww, c);

				}
			}
		}
	}
	return;
}

void ReLULayer::backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads, const Param& param) {


	// 1. 设置输出梯度Blob的尺寸 (dx----grdas[0])
	grads[0].reset(new Blob(*cache[0]));

	// 2. 获取掩码mask
	int N = grads[0]->getN();
	for (int n = 0; n < N; n++) // 输出cube数
		(*grads[0])[n].transform([](double e) {return e > 0 ? 1 : 0; });
	(*grads[0]) = (*grads[0]) * (*din);
	return;
}

void ConvLayer::backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads, const Param& param) {

	// 1. 设置输出梯度Blob的尺寸 (dx----grdas[0])
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));
	grads[1].reset(new Blob(cache[1]->size(), TZEROS));
	grads[2].reset(new Blob(cache[2]->size(), TZEROS));
	// 2. 获取输入梯度Blob的尺寸
	int Nd = din->getN();        //输入梯度Blob中cube个数（该batch样本个数）
	int Cd = din->getC();         //输入梯度Blob通道数
	int Hd = din->getH();      //输入梯度Blob高
	int Wd = din->getW();    //输入梯度Blob宽
	// 3. 获取卷积核相关参数
	int Hw = param.conv_height;
	int Ww = param.conv_width;
	int stride = param.conv_stride;

	// 4. 开始反向传播
	Blob padX = cache[0]->pad(param.conv_pad);
	Blob pad_dx(padX.size(), TZEROS);
	for (int n = 0; n < Nd; n++) { // 输出cube数
		for (int c = 0; c < Cd; c++) { // 输出通道数
			for (int hh = 0; hh < Hd; hh++) { // 输出Blob的高
				for (int ww = 0; ww < Wd; ww++) { // 输出Blob的宽
					// (1)获取掩码mask
					cube window = padX[n](span(hh * stride, hh * stride + Hw - 1), span(ww * stride, ww * stride + Ww - 1), span::all);
					// dx
					pad_dx[n](span(hh * stride, hh * stride + Hw - 1), span(ww * stride, ww * stride + Ww - 1), span::all) += (*din)[n](hh, ww, c) * (*cache[1])[c];
					// dw
					(*grads[1])[c] += (*din)[n](hh, ww, c) * window / Nd;
					// db
					(*grads[2])[c](0, 0, 0) += (*din)[n](hh, ww, c) / Nd;
				}
			}
		}
	}
	// 去掉输出梯度中的padding部分
	(*grads[0]) = pad_dx.unPad(param.conv_pad);
	return;
}