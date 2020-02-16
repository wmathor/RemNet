#include "myNet.hpp"
#include <json/json.h>
#include <fstream>
#include <cassert>
using namespace std;

void NetParam::readNetParam(string file) {
	ifstream ifs(file);
	assert(ifs.is_open()); // 确保文件正确打开
	Json::CharReaderBuilder reader;
	JSONCPP_STRING errs;
	Json::Value value;     // 存储器
	if (Json::parseFromStream(reader, ifs, &value, &errs)) {
		
		if (!value["train"].isNull()) {
			auto& tparam = value["train"]; // 通过引用方式拿到"train"对象里面的所有元素
			this->lr = tparam["learning rate"].asDouble(); // 解析成Double类型
			this->lr_decay = tparam["lr decay"].asDouble();
			this->optimizer= tparam["update method"].asString(); // 解析成String类型
			this->momentum = tparam["momentum parameter"].asDouble();
			this->epochs = tparam["epochs"].asInt(); // 解析成Int类型
			this->use_batch = tparam["use batch"].asBool(); // 解析成Bool类型
			this->batch_size = tparam["batch size"].asInt();
			this->acc_frequence = tparam["acc frequence"].asInt();
			this->update_lr = tparam["frequence update"].asBool();
			this->snap_shot = tparam["snapshot"].asBool();
			this->snapshot_interval = tparam["snapshot interval"].asInt();
			this->fine_tune = tparam["fine tune"].asBool();
			this->preTrainedModel = tparam["pre trained model"].asString();
		}

		if (!value["net"].isNull()) {
			auto& nparam = value["net"];
			for (int i = 0; i < (int)nparam.size(); i++) {
				auto& layer = nparam[i];
				std::string name = layer["name"].asString();

				this->layers.push_back(name);
				this->ltypes.push_back(layer["type"].asString());

				if (layer["type"].asString() == "Conv") {
					int num = layer["kernel num"].asInt();
					int width = layer["kernel width"].asInt();
					int height = layer["kernel height"].asInt();
					int pad = layer["pad"].asInt();
					int stride = layer["stride"].asInt();

					this->lparams[name].conv_kernels = num;
					this->lparams[name].conv_height = height;
					this->lparams[name].conv_width = width;
					this->lparams[name].conv_pad = pad;
					this->lparams[name].conv_stride = stride;
				}

				if (layer["type"].asString() == "Pool") {
					int width = layer["kernel width"].asInt();
					int height = layer["kernel height"].asInt();
					int stride = layer["stride"].asInt();

					this->lparams[name].pool_height = height;
					this->lparams[name].pool_width = width;
					this->lparams[name].pool_stride = stride;
				}

				if (layer["type"].asString() == "FC") {
					int num = layer["kernel num"].asInt();
					
					this->lparams[name].fc_kernels = num;
				}
			}
		}
	}	
}

void Net::initNet(NetParam& param, vector<shared_ptr<Blob>>& x, vector<shared_ptr<Blob>>& y) {
	// 1. 打印层结构
	layers = param.layers;
	ltypes = param.ltypes;
	for (int i = 0; i < layers.size(); i++)
		cout << "layer = " << layers[i] << "," << "ltypes = " << ltypes[i] << endl;

	// 2. 初始化Net类中的成员变量
	x_train = x[0];
	y_train = y[0];
	x_val = x[1];
	y_val = y[1];

	for (int i = 0; i < (int)layers.size(); i++) { // 遍历每一层
		data[layers[i]] = vector<shared_ptr<Blob>>(3, NULL); //x, w, b
		gradient[layers[i]] = vector<shared_ptr<Blob>>(3, NULL);
		outShapes[layers[i]] = vector<int>(4); // 定义缓存，存储每一层的输出尺寸
	}

	 // 3. 完成每一层w和b的初始化
	shared_ptr<Layer> myLayer(NULL);
	vector<int> inShape = {param.batch_size, x_train->getC(), x_train->getH(), x_train->getW()};
	cout << "input -> (" << inShape[0] << ", " << inShape[1] << ", " << inShape[2] << ", " << inShape[3] << ")" << endl;
	for (int i = 0; i < (int)layers.size() - 1; i++) {
		string lname = layers[i];
		string ltype = ltypes[i];
		// conv1 -> relu1 -> pool1 -> fc -> softmax
		if (ltype == "Conv") {
			myLayer.reset(new ConvLayer);
		}
		if (ltype == "ReLU") {
			myLayer.reset(new ReLULayer);
			
		}
		if (ltype == "Pool") {
			myLayer.reset(new PoolLayer);
		}
		if (ltype == "FC") {
			myLayer.reset(new FCLayer);
		}
		myLayers[lname] = myLayer;
		myLayer->initLayer(inShape, lname, data[lname], param.lparams[lname]);
		myLayer->calcShape(inShape, outShapes[lname], param.lparams[lname]);
		inShape.assign(outShapes[lname].begin(), outShapes[lname].end());
		cout << lname << "->(" << outShapes[lname][0] << "," << outShapes[lname][1] << "," << outShapes[lname][2] << "," << outShapes[lname][3] << ")" << endl;
	}
}

void Net::trainNet(NetParam& param) {
	
	int N = x_train->getN(); // 样本总数
	int iter_per_epoch = N / param.batch_size;
	// 总的批次数（迭代次数）=单个epoch所含的批次数 * epoch个数
	int batchs = iter_per_epoch * param.epochs;
	for (int iter = 0; iter < 200; iter++) {
		// 1. 从整个训练集中获取一个mini-batch
		shared_ptr<Blob> x_batch;
		shared_ptr<Blob> y_batch;
		x_batch.reset(new Blob(
			(x_train->subBlob(
			(iter*param.batch_size) % N,
			(((iter+1)*param.batch_size) % N)
		))));

		y_batch.reset(new Blob(
			(y_train->subBlob(
			(iter * param.batch_size) % N,
				(((iter + 1) * param.batch_size) % N)
			))));

		// 2. 用该mini-batch训练网络模型
		train_with_batch(x_batch, y_batch, param);

		// 3. 评估模型当前准确率（训练集和验证集）
		evaluate_with_batch(param);
		printf("iter: %d   lr: %0.6f   loss: %f   train_acc: %0.2f%%   val_acc: %0.2f%%\n",
			iter, param.lr, loss_, train_accu * 100, val_accu * 100);
	}
}

void Net::train_with_batch(shared_ptr<Blob> &x, shared_ptr<Blob>& y, NetParam& param, string mode) {

	// 1. 将mini-batch填充到初始层的x当中
	data[layers[0]][0] = x;
	data[layers.back()][1] = y;

	// 2. 逐层前向计算 conv1->relu1->pool1->fc1->softmax
	int n = layers.size(); // 层数
	for (int i = 0; i < n - 1; i++) {
		string lname = layers[i];
		shared_ptr<Blob> out;
		myLayers[lname]->forward(data[lname], out, param.lparams[lname]);
		data[layers[i+1]][0] = out;
	
	}
	if (mode == "TEST") // 如果仅用于前向传播做测试，则不需要参数更新
		return;
	// 3. softmax 和 计算loss
	SoftmaxLossLayer::softmax_cross_entropy_with_logits(data[layers.back()], loss_, gradient[layers.back()][0]);
	// 4. 逐层反向传播 conv1<-relu1<-pool1<-fc1<-softmax
	for (int i = n - 2; i >= 0; i--) {
		string lname = layers[i];
		myLayers[lname]->backward(gradient[layers[i+1]][0], data[lname], gradient[lname], param.lparams[lname]);
	}
	// 5. 参数更新（利用梯度下降）
	optimizer_with_batch(param);
}

Blob& Blob::operator= (double val) {
	for (int i = 0; i < N; i++)
		blob_data[i].fill(val);
	return *this;
}

void Net::optimizer_with_batch(NetParam& param) {
	for (auto lname : layers) {
		// 跳过没有weight和bias的层
		if (!data[lname][1] || !data[lname][2])
			continue;

		for (int i = 1; i <= 2; i++) {
			assert(param.optimizer == "sgd" || param.optimizer == "momentum" || param.optimizer == "rmsprop");
			shared_ptr<Blob> dparam(new Blob(data[lname][i]->size(), TZEROS));
			(*dparam) = -param.lr * (*gradient[lname][i]);
			(*data[lname][i]) = (*data[lname][i]) + (*dparam);
		}
	}
	// update lr
	if (param.update_lr)
		param.lr *= param.lr_decay;
}

void Net::evaluate_with_batch(NetParam& param) {
	// 评估训练集准确率
	shared_ptr<Blob> x_train_subset;
	shared_ptr<Blob> y_train_subset;
	int N = x_train->getN();
	if (N > 1000) {
		x_train_subset.reset(new Blob(x_train->subBlob(0, 1000)));
		y_train_subset.reset(new Blob(y_train->subBlob(0, 1000)));
	} else {
		x_train_subset = x_train;
		y_train_subset = y_train;
	}
	train_with_batch(x_train_subset, y_train_subset, param, "TEST");
	train_accu = calc_accuracy(*data[layers.back()][1], *data[layers.back()][0]);
	// 评估验证集准确率
	
	train_with_batch(x_val, y_val, param, "TEST");
	val_accu = calc_accuracy(*data[layers.back()][1], *data[layers.back()][0]);



}

double Net::calc_accuracy(Blob& y, Blob& pred) {
	vector<int> size_y = y.size();
	vector<int> size_p = pred.size();
	for (int i = 0; i < 4; i++)
		assert(size_y[i] == size_p[i]);
	// 遍历所有cube，找出y和pred的最大值所对应的下标，进行比对
	int N = y.getN();
	int count = 0; // 正确个数
	for (int n = 0; n < N; n++)
		if (y[n].index_max() == pred[n].index_max())
			count++;
	return (double)count / (double)N; // acc%
}