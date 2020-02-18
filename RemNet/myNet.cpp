#include "myNet.hpp"
#include "myBlob.hpp"
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
			this->rmsprop = tparam["rmsprop"].asDouble();
			this->reg = tparam["reg coefficient"].asDouble();
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
					this->lparams[name].conv_kernels = layer["kernel num"].asInt();
					this->lparams[name].conv_height = layer["kernel height"].asInt();
					this->lparams[name].conv_width = layer["kernel width"].asInt();
					this->lparams[name].conv_pad = layer["pad"].asInt();
					this->lparams[name].conv_stride = layer["stride"].asInt();
					this->lparams[name].conv_weight_init = layer["conv weight init"].asString();
				}

				if (layer["type"].asString() == "Pool") {
					this->lparams[name].pool_height = layer["kernel height"].asInt();
					this->lparams[name].pool_width = layer["kernel width"].asInt();
					this->lparams[name].pool_stride = layer["stride"].asInt();
				}

				if (layer["type"].asString() == "FC") {
					this->lparams[name].fc_kernels = layer["kernel num"].asInt();
					this->lparams[name].fc_weight_init = layer["fc weight init"].asString();
				}

				if (layer["type"].asString() == "Dropout") {
					this->lparams[name].drop_rate = layer["drop rate"].asDouble();
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
		step_cache[layers[i]] = vector<shared_ptr<Blob>>(3, NULL);
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
		if (ltype == "Dropout") {
			myLayer.reset(new DropoutLayer);
		}
		myLayers[lname] = myLayer;
		myLayer->initLayer(inShape, lname, data[lname], param.lparams[lname]);
		myLayer->calcShape(inShape, outShapes[lname], param.lparams[lname]);
		inShape.assign(outShapes[lname].begin(), outShapes[lname].end());
		cout << lname << "->(" << outShapes[lname][0] << "," << outShapes[lname][1] << "," << outShapes[lname][2] << "," << outShapes[lname][3] << ")" << endl;
	}
	if (param.fine_tune) {
		fstream input(param.preTrainedModel, ios::in | ios::binary);
		if (!input) {
			cout << param.preTrainedModel << "was not found !" << endl;
			return;
		}
		shared_ptr<RemNet::snapshotModel> snapshot_model(new RemNet::snapshotModel);
		if (!snapshot_model->ParseFromIstream(&input)) {
			cout << "Failed to parse the " << param.preTrainedModel << endl;
			return;
		}
		cout << "-----Load the " << param.preTrainedModel << " successfully !" << endl;
		loadModelParam(snapshot_model);
	}
}

void Net::trainNet(NetParam& param) {
	
	int N = x_train->getN(); // 样本总数
	int iter_per_epoch = N / param.batch_size;
	// 总的批次数（迭代次数）=单个epoch所含的批次数 * epoch个数
	int batchs = iter_per_epoch * param.epochs;
	for (int iter = 0; iter < batchs; iter++) {
		// 1. 从整个训练集中获取一个mini-batch
		shared_ptr<Blob> x_batch;
		shared_ptr<Blob> y_batch;
		x_batch.reset(new Blob(
			(x_train->subBlob(
			(iter * param.batch_size) % N,
			(((iter +1)*param.batch_size) % N)
		))));

		y_batch.reset(new Blob(
			(y_train->subBlob(
			(iter * param.batch_size) % N,
				(((iter + 1) * param.batch_size) % N)
			))));

		// 2. 用该mini-batch训练网络模型
		train_with_batch(x_batch, y_batch, param);

		// 3. 评估模型当前准确率（训练集和验证集）
		if (iter % param.acc_frequence == 0) {
			evaluate_with_batch(param);
			printf("iter_%d   lr: %0.6f   train_loss: %f   val_loss: %f   train_acc: %0.2f%%   val_acc: %0.2f%%\n",
				iter, param.lr, train_loss, val_loss, train_accu * 100, val_accu * 100);
		}
		// 4. 保存模型快照
		if (iter > 0 && param.snap_shot && iter % param.snapshot_interval == 0) {
			// (1) 定义输出文件
			char outputFile[40];
			sprintf_s(outputFile, "./iter%d.RemNetModel", iter);
			fstream output(outputFile, ios::out | ios::trunc | ios::binary);

			// (2) 把Blob中的参数保存到snapshotModel中
			shared_ptr<RemNet::snapshotModel> snapshot_model(new RemNet::snapshotModel);
			saveModelParam(snapshot_model);

			// (3) 调用SerializeToOstream()函数将snapshotModel中的数据写成一个二进制文件
			if (!snapshot_model->SerializeToOstream(&output)) {
				cout << "Failed to Serialize snapshot_model to Ostream";
				return;
			}

		}
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
		myLayers[lname]->forward(data[lname], out, param.lparams[lname], mode);
		data[layers[i+1]][0] = out;
	
	}
	if (mode == "TRAIN") {
		// 3. softmax 和 计算loss
		if (ltypes.back() == "Softmax")
			SoftmaxLossLayer::softmax_cross_entropy_with_logits(data[layers.back()], train_loss, gradient[layers.back()][0]);
		if (ltypes.back() == "SVM")
			SVMLossLayer::hinge_with_logits(data[layers.back()], train_loss, gradient[layers.back()][0]);
	} else {
		if (ltypes.back() == "Softmax")
			SoftmaxLossLayer::softmax_cross_entropy_with_logits(data[layers.back()], val_loss, gradient[layers.back()][0]);
		if (ltypes.back() == "SVM")
			SVMLossLayer::hinge_with_logits(data[layers.back()], val_loss, gradient[layers.back()][0]);
	}
	if (mode == "TRAIN") {
		// 4. 逐层反向传播 conv1<-relu1<-pool1<-fc1<-softmax
		for (int i = n - 2; i >= 0; i--) {
			string lname = layers[i];
			myLayers[lname]->backward(gradient[layers[i + 1]][0], data[lname], gradient[lname], param.lparams[lname]);
		}
	}

	// 5. 对各层梯度施加L2正则化的影响
	if (param.reg != 0)
		regular_with_batch(param, mode);

	// 6. 参数更新（利用梯度下降）
	if (mode == "TRAIN")
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
			if (param.optimizer == "rmsprop") {
				double rmsprop = param.rmsprop;
				if (!step_cache[lname][i])
					step_cache[lname][i].reset(new Blob(data[lname][i]->size(), TZEROS));
				(*step_cache[lname][i]) = rmsprop * (*step_cache[lname][i]) + (1 - rmsprop) * (*gradient[lname][i]) * (*gradient[lname][i]);
				(*dparam) = -param.lr * (*gradient[lname][i]) / sqrt((*step_cache[lname][i]) + 1e-8);
			}
				
			else if (param.optimizer == "momentum") {
				if (!step_cache[lname][i])
					step_cache[lname][i].reset(new Blob(data[lname][i]->size(), TZEROS));
				(*step_cache[lname][i]) = param.momentum * (*step_cache[lname][i]) + (*gradient[lname][i]);
				(*dparam) = -param.lr * (*step_cache[lname][i]);
			}
			else
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

void Net::saveModelParam(shared_ptr<RemNet::snapshotModel>& snapshot_model) {
	// 没有weight和bias的不需要存储
	for (auto lname : layers) {
		if (!data[lname][1] || !data[lname][2])
			continue;
		// 取出相关Blob中的所有参数，填入snapshotModel中
		for (int i = 1; i <= 2; i++) { // weight和bias
			RemNet::snapshotModel_paramBlok* param_blok = snapshot_model->add_param_blok(); // 动态添加一个param_blok
			int N = data[lname][i]->getN();
			int C = data[lname][i]->getC();
			int H = data[lname][i]->getH();
			int W = data[lname][i]->getW();
			param_blok->set_kernel_n(N);
			param_blok->set_kernel_c(C);
			param_blok->set_kernel_h(H);
			param_blok->set_kernel_w(W);
			param_blok->set_layer_name(lname);
			if (i == 1)
				param_blok->set_param_type("WEIGHT");
			else
				param_blok->set_param_type("BIAS");
			for (int n = 0; n < N; n++) {
				for (int c = 0; c < C; c++) {
					for (int h = 0; h < H; h++) {
						for (int w = 0; w < W; w++) {
							RemNet::snapshotModel_paramBlok_paramValue* param_value = param_blok->add_param_value();
							param_value->set_value((*data[lname][i])[n](h, w, c));
						}
					}
				}
			}
		}
	}
}

void Net::loadModelParam(const shared_ptr<RemNet::snapshotModel>& snapshot_model) {
	for (int i = 0; i < snapshot_model->param_blok_size(); i++) {
		// 1. 从snapshot_model中逐一取出paramBlok
		const RemNet::snapshotModel::paramBlok& param_blok = snapshot_model->param_blok(i);
		
		// 2. 取出paramBlok中的标记型变量
		string lname = param_blok.layer_name();
		string paramtype = param_blok.param_type();
		int N = param_blok.kernel_n();
		int C = param_blok.kernel_c();
		int H = param_blok.kernel_h();
		int W = param_blok.kernel_w();
		cout << lname << ":" << paramtype << ": (" << N << ", " << C << ", " << H << ", " << W << ")" << endl;

		// 3. 遍历当前paramBlok中的每一个参数，取出，填入对应的Blob中
		int val_idx = 0;
		shared_ptr<Blob> tmp_blob(new Blob(N, C, H, W));
		for (int n = 0; n < N; n++) {
			for (int c = 0; c < C; c++) {
				for (int h = 0; h < H; h++) {
					for (int w = 0; w < W; w++) {
						const RemNet::snapshotModel_paramBlok_paramValue&param_value = param_blok.param_value(val_idx++);
						(*tmp_blob)[n](h, w, c) = param_value.value();
					}
				}
			}
		}

		// 4. 将tmp_blob赋值到data中
		if (paramtype == "WEIGHT")
			data[lname][1] = tmp_blob;
		else
			data[lname][2] = tmp_blob;
	}
}

void Net::regular_with_batch(NetParam& param, string mode) {
	double reg_loss = 0;
	int N = data[layers[0]][0]->getN();
	for (auto lname : layers) {
		if (!gradient[lname][1])
			continue;
		if (mode == "TRAIN")
			(*gradient[lname][1]) = (*gradient[lname][1]) + param.reg * (*data[lname][1]) / N;
		reg_loss += accu(square((*data[lname][1])));
	}
	reg_loss = reg_loss * param.reg / (N << 1);
	if (mode == "TRAIN")
		train_loss = train_loss + reg_loss;
	else
		val_loss = val_loss + reg_loss;
}