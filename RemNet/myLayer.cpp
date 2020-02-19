#include "myLayer.hpp"
#include <cassert>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace arma;


// arma::mat converted to cv::mat
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
	// 1. Get conv kernel shape (F, C, H, W)
	int tF = param.conv_kernels;
	int tC = inShape[1];
	int tH = param.conv_height;
	int tW = param.conv_width;
	
	// 2. Initializes the Blob that stores weight and bias (in[1], in[2]) = (w, b)
	if (!in[1]) { // The blob that stores weight is not empty
		in[1].reset(new Blob(tF, tC, tH, tW, TRANDN));
		if (param.conv_weight_init == "msra")
			(*in[1]) *= std::sqrt(2 / (double)(inShape[1] * inShape[2] * inShape[3]));
		else
			(*in[1]) *= 1e-2;
	}
	if (!in[2]) { // The blob that stores bias is not empty
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
		// 1. Get FC shape(F, C, H, W)
		int tF = param.fc_kernels;
		int tC = inShape[1];
		int tH = inShape[2];
		int tW = inShape[3];

		// 2. Initializes the Blob that stores weight and bias (in[1], in[2]) = (w, b)
		if (!in[1]) { // The blob that stores weight is not empty
			in[1].reset(new Blob(tF, tC, tH, tW, TRANDN));
			if (param.fc_weight_init == "msra")
				(*in[1]) *= std::sqrt(2 / (double)(inShape[1] * inShape[2] * inShape[3]));
			else
				(*in[1]) *= 1e-2;
		}
		if (!in[2])// The blob that stores bias is not empty
			in[2].reset(new Blob(tF, 1, 1, 1, TZEROS));
		return;
}
///////////////////////////////////calcShape/////////////////////////////////////////
void ConvLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	// 1. Get input Blob shape
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];
	
	// 2. Get conv kernel shape
	int tF = param.conv_kernels; // kernel numers
	int tH = param.conv_height;  // kernel height
	int tW = param.conv_width;   // kernel width
	int tP = param.conv_pad;     // kernel padding
	int tS = param.conv_stride;  // kernel stride

	// 3. Calc conved shape
	int No = Ni;
	int Co = tF;
	int Ho = (Hi + (tP << 1) - tH) / tS + 1;
	int Wo = (Wi + (tP << 1) - tW) / tS + 1;

	// 4. Assign the size of the output Blob
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}

void ReLULayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	outShape.assign(inShape.begin(), inShape.end()); // Copy inShape to outShape (deep copy)
	return;
}

void PoolLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	// 1. Get input Blob shape
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];

	// 2. get conv kernel shape
	int tH = param.pool_height;  // kernel height
	int tW = param.pool_width;   // kernel width
	int tS = param.pool_stride;  // kernel stride

	// 3. Calc pooled shape
	int No = Ni;
	int Co = Ci;
	int Ho = (Hi - tH) / tS + 1;
	int Wo = (Wi - tW) / tS + 1;

	// 4. Assign the size of the output Blob
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}
void FCLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	// 1. Get input Blob shape
	int No = inShape[0]; // batch size
	int Co = param.fc_kernels; // current layer nn numbers
	int Ho = 1;
	int Wo = 1;

	// 2. Assign the size of the output Blob
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}
///////////////////////////////////forward///////////////////////////////////
void ConvLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode) {
	if (out)
		out.reset();
	// 1. Get related parameters（input, conv kernel, output）
	assert(in[0]->getC() == in[1]->getC());
	int N = in[0]->getN();   // The number of cubes in the input Blob
	int C = in[0]->getC();   // The number of channels in the input Blob
	int Hx = in[0]->getH();  // input Blob height
	int Wx = in[0]->getW();  // input Blob width

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
	//visiable((*out)[0], vec_mat_out); // Visualize the first convolution kernel
 	return;
}

void ReLULayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode) {
	if (out)
		out.reset();
	out.reset(new Blob(*in[0]));
	out->maxIn(0);
	return;
}

void PoolLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode) {
	if (out)
		out.reset();
	// 1. Get related parameters (input, pooling kernel, output)
	int N = in[0]->getN();   // The number of cubes in the input Blob
	int C = in[0]->getC();   // The number of channels in the input Blob
	int Hx = in[0]->getH();  // input Blob height
	int Wx = in[0]->getW();  // input Blob width

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

void FCLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode) {



	if (out)
		out.reset();
	// 1. Get related parameters (input, full connection kernel, output)

	int N = in[0]->getN();   // The number of cubes in the input Blob
	int C = in[0]->getC();   // The number of channels in the input Blob
	int Hx = in[0]->getH();  // input Blob height
	int Wx = in[0]->getW();  // input Blob width

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

	// 1. Get related parameters 
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
		// Gradient expression derivation
		(*dout)[i] = prob - (*in[1])[i]; // Calculate the error signal generated by each sample (reverse gradient)

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

	// 1. Set the size of the output gradient Blob (dx = grdas[0])
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));
	// 2. Gets the size of the input gradient Blob
	int Nd = din->getN();        // Number of cubes in input gradient Blob (number of batch samples)
	int Cd = din->getC();        // Number of channels for input gradient Blob
	int Hd = din->getH();        // Number of height for input gradient Blob
	int Wd = din->getW();        // Number of width for input gradient Blob
	// 3. Gets the parameters associated with the pooling kernel
	int Hp = param.pool_height;
	int Wp = param.pool_width;
	int stride = param.pool_stride;

	// 4. Start backward
	for (int n = 0; n < Nd; n++) { // The output cubes number
		for (int c = 0; c < Cd; c++) { // The output channels number
			for (int hh = 0; hh < Hd; hh++) { // The height of the output Blob
				for (int ww = 0; ww < Wd; ww++) { // The width of the output Blob
					// (1) get mask
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


	// 1. Set the size of the output gradient Blob (dx = grdas[0])
	grads[0].reset(new Blob(*cache[0]));

	// 2. get mask
	int N = grads[0]->getN();
	for (int n = 0; n < N; n++) {// The output cube number
		//(*grads[0])[n].transform([](double e) {return e > 0 ? 1 : 0; });
		(*grads[0])[n].transform([](double e) {return e < 6 ? 1 : 0; });
	}
	(*grads[0]) = (*grads[0]) * (*din);
	return;
}

void ConvLayer::backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads, const Param& param) {

	// 1. Set the size of the output gradient Blob (dx = grdas[0])
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));
	grads[1].reset(new Blob(cache[1]->size(), TZEROS));
	grads[2].reset(new Blob(cache[2]->size(), TZEROS));
	// 2. Gets the size of the input gradient Blob
	int Nd = din->getN();        // Number of cubes in input gradient Blob (number of batch samples)
	int Cd = din->getC();        // Enter the number of gradient Blob channels
	int Hd = din->getH();        // Input gradient Blob height
	int Wd = din->getW();        // Input gradient Blob width
	// 3. Get the convolution kernel correlation parameters
	int Hw = param.conv_height;
	int Ww = param.conv_width;
	int stride = param.conv_stride;

	// 4. start backward
	Blob padX = cache[0]->pad(param.conv_pad);
	Blob pad_dx(padX.size(), TZEROS);
	for (int n = 0; n < Nd; n++) {
		for (int c = 0; c < Cd; c++) {
			for (int hh = 0; hh < Hd; hh++) {
				for (int ww = 0; ww < Wd; ww++) {
					// (1) get mask
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
	// Remove the padding from the output gradient
	(*grads[0]) = pad_dx.unPad(param.conv_pad);
	return;
}

void SVMLossLayer::hinge_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout) {
	if (dout)
		dout.reset();

	// 1. Get relevant dimensions
	int N = in[0]->getN();
	int C = in[0]->getC();
	int Hx = in[0]->getH();
	int Wx = in[0]->getW();
	assert(Hx == 1 && Wx == 1);

	dout.reset(new Blob(N, C, Hx, Wx)); // (N, C, 1, 1)
	double loss_ = 0;
	double delta = 0.2;
	for (int i = 0; i < N; i++) {
		// Calc Loss
		int idx_max = (*in[1])[i].index_max();
		double positive_x = (*in[0])[i](0, 0, idx_max);
		cube tmp = ((*in[0])[i] - positive_x + delta); // Hinge Loss formula
		tmp(0, 0, idx_max) = 0; // Eliminate values in the correct category
		tmp.transform([](double e) {return e > 0 ? e : 0; });
		arma::accu(tmp); // get all kinds of losses
		loss_ += arma::accu(tmp);
		
		// Calc Gradient
		tmp.transform([](double e) {return e ? 1 : 0; });
		tmp(0, 0, idx_max) = -arma::accu(tmp);
		(*dout)[i] = tmp;
	}
	loss = loss_ / N;
	return;
}

void DropoutLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param) {
	return;
}
void DropoutLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	outShape.assign(inShape.begin(), inShape.end());
	return;
}
void DropoutLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode) {
	if (out)
		out.reset();
	if (mode == "TRAIN") {
		double drop_rate = param.drop_rate;
		assert(drop_rate >= 0 && drop_rate <= 1);
		shared_ptr<Blob> in_mask(new Blob(in[0]->size(), TRANDU));
		in_mask->convertIn(drop_rate);
		drop_mask.reset(new Blob(*in_mask));
		out.reset(new Blob((*in[0]) * (*in_mask) / (1 - drop_rate)));
	}
	else
		out.reset(new Blob((*in[0])));
}
void DropoutLayer::backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads, const Param& param) {
	double drop_rate = param.drop_rate;
	grads[0].reset(new Blob((*din) * (*drop_mask) / (1 - drop_rate)));
}

void BNLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param) {
	int C = inShape[1];
	int H = inShape[2];
	int W = inShape[3];
	if (!in[1]) 
		in[1].reset(new Blob(1, C, H, W, TZEROS));
	if (!in[2])
		in[2].reset(new Blob(1, C, H, W, TZEROS));
}


void BNLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode) {
	if (out)
		out.reset(new Blob(in[0]->size(), TZEROS));
	int N = in[0]->getN();
	int C = in[0]->getC();
	int H = in[0]->getH();
	int W = in[0]->getW();

	if (mode == "TRAIN") {
		// clear
		mean.reset(new cube(1, 1, C, fill::zeros));
		var.reset(new cube(1, 1, C, fill::zeros));
		std.reset(new cube(1, 1, C, fill::zeros));

		// calc mean
		for (int i = 0; i < N; i++)
			(*mean) += sum(sum((*in[0])[i], 0), 1) / (H * W);
		(*mean) /= (-N);
		
		// calc variance
		for (int i = 0; i < N; i++)
			(*var) += square(sum(sum((*in[0])[i], 0), 1) / (H * W) + (*mean));
		(*var) /= N;

		// calc std
		(*std) = sqrt((*var) + 1e-5);
		
		// broadcast mean and std
		cube mean_tmp(H, W, C, fill::zeros);
		cube std_tmp(H, W, C, fill::zeros);
		for (int c = 0; c < C; c++) {
			mean_tmp.slice(c).fill(as_scalar((*mean).slice(c)));
			std_tmp.slice(c).fill(as_scalar((*std).slice(c)));
		}

		// normalize
		for (int i = 0; i < N; i++)
			(*out)[i] = ((*in[0])[i] + mean_tmp) / std_tmp;

		if (!running_mean_std_init) {
			(*in[1])[0] = mean_tmp;
			(*in[2])[0] = std_tmp;
			running_mean_std_init = true;
		}

		double yita = 0.99;
		(*in[1])[0] = yita * (*in[1])[0] + (1 - yita) * mean_tmp;
		(*in[2])[0] = yita * (*in[2])[0] + (1 - yita) * std_tmp;
	} else 
		for (int n = 0; n < N; n++)
			(*out)[n] = ((*in[0])[n] + (*in[1])[0]) / (*in[2])[0];
}

void BNLayer::backward(const shared_ptr<Blob>& din, const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads, const Param& param) {
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));
	int N = grads[0]->getN();
	int C = grads[0]->getC();
	int H = grads[0]->getH();
	int W = grads[0]->getW();

	cube mean_tmp(H, W, C, fill::zeros);
	cube var_tmp(H, W, C, fill::zeros);
	cube std_tmp(H, W, C, fill::zeros);
	for (int c = 0; c < C; c++) {
		mean_tmp.slice(c).fill(as_scalar((*mean).slice(c)));
		var_tmp.slice(c).fill(as_scalar((*var).slice(c)));
		std_tmp.slice(c).fill(as_scalar((*std).slice(c)));
	}

	for (int k = 0; k < N; k++) {
		cube item1(H, W, C, fill::zeros);
		for (int i = 0; i < N; i++)
			item1 += (*din)[i] % ((*cache[0])[i] + mean_tmp);
		cube tmp = (-sum(sum(item1, 0), 1) / (2 * (*var) % (*std))) / N;

		cube item2(1, 1, C, fill::zeros);
		for (int i = 0; i < N; i++)
			item2 += (tmp % (2 * (sum(sum((*cache[0])[i], 0), 1) / (H * W) + (*mean))));

		cube item3(H, W, C, fill::zeros);
		for (int i = 0; i < N; i++)
			item2 += (*din)[i] / std_tmp;
		
		cube item4(1, 1, C, fill::zeros);
		item4 = sum(sum(item3, 0), 1);

		cube black0 = (item2 + item4) / (-N);
		cube red0 = (tmp % (2 * (sum(sum((*cache[0])[k], 0), 1) / (H * W) + (*mean))));
		cube black_(H, W, C, fill::zeros);
		cube red_(H, W, C, fill::zeros);
		cube purple_ = (*din)[k] / std_tmp;
		for (int c = 0; c < C; ++c) {
			black_.slice(c).fill(as_scalar(black0.slice(c)));        //cube(H, W, C)
			red_.slice(c).fill(as_scalar(red0.slice(c)));			//cube(H, W, C)
		}
		(*grads[0])[k] = (black_ + red_) / (H * W) + purple_;
	}
	return;
}

void BNLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	outShape.assign(inShape.begin(), inShape.end());
	return;
}

void ScaleLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param) {
	int C = inShape[1];

	if (!in[1])
		in[1].reset(new Blob(1, C, 1, 1, TONES));

	if (!in[2])
		in[2].reset(new Blob(1, C, 1, 1, TZEROS));
	return;
}

void ScaleLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	outShape.assign(inShape.begin(), inShape.end());
	return;
}

void ScaleLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode) {
	out.reset(new Blob(in[0]->size(), TZEROS));

	int N = in[0]->getN();
	int C = in[0]->getC();
	int H = in[0]->getH();
	int W = in[0]->getW();

	shared_ptr<Blob> gamma(new Blob(1, C, H, W, TZEROS));
	shared_ptr<Blob> beta(new Blob(1, C, H, W, TZEROS));
	for (int c = 0; c < C; ++c) {
		(*gamma)[0].slice(c).fill(as_scalar((*in[1])[0].slice(c)));
		(*beta)[0].slice(c).fill(as_scalar((*in[2])[0].slice(c)));
	}

	for (int n = 0; n < N; ++n)
		(*out)[n] = (*gamma)[0] % (*in[0])[n] + (*beta)[0];  //out  = γ * in    +  β
	return;
}

void ScaleLayer::backward(const shared_ptr<Blob>& din,
	const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads,
	const Param& param) {

	grads[0].reset(new Blob(cache[0]->size(), TZEROS));//dx  
	grads[1].reset(new Blob(cache[1]->size(), TZEROS));//dγ
	grads[2].reset(new Blob(cache[2]->size(), TZEROS));//dβ
	int N = grads[0]->getN();
	int C = grads[0]->getC();
	int H = grads[0]->getH();
	int W = grads[0]->getW();

	shared_ptr<Blob> gamma(new Blob(1, C, H, W, TZEROS));
	for (int c = 0; c < C; ++c)
		(*gamma)[0].slice(c).fill(as_scalar((*cache[1])[0].slice(c)));  //因为dx  = din % γ  ，所以γ 需要广播完成尺寸匹配

	shared_ptr<Blob> dgamma(new Blob(1, C, H, W, TZEROS));
	shared_ptr<Blob> dbeta(new Blob(1, C, H, W, TZEROS));
	for (int n = 0; n < N; ++n) {
		(*grads[0])[n] = (*din)[n] % (*gamma)[0];
		(*dgamma)[0] += (*din)[n] % (*cache[0])[n];
		(*dbeta)[0] += (*din)[n];
	}
	(*grads[1])[0] = sum(sum((*dgamma)[0], 0), 1) / N;
	(*grads[2])[0] = sum(sum((*dbeta)[0], 0), 1) / N;

	return;
}

void TanhLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param) {
	return;
}

void TanhLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) {
	outShape.assign(inShape.begin(), inShape.end());
	return;
}

void TanhLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode) {
	if (out)
		out.reset();
	out.reset(new Blob(*in[0]));
	int N = in[0]->getN();
	for (int n = 0; n < N; ++n)
		(*out)[n] = (arma::exp((*in[0])[n]) - arma::exp(-(*in[0])[n])) / (arma::exp((*in[0])[n]) + arma::exp(-(*in[0])[n]));
	return;
}

void TanhLayer::backward(const shared_ptr<Blob>& din,
	const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads,
	const Param& param) {

	grads[0].reset(new Blob(*cache[0]));

	int N = grads[0]->getN();
	for (int n = 0; n < N; ++n)
		(*grads[0])[n] = (*din)[n] % (1 - arma::square((arma::exp((*cache[0])[n]) - arma::exp(-(*cache[0])[n])) / (arma::exp((*cache[0])[n]) + arma::exp(-(*cache[0])[n]))));
	return;
}