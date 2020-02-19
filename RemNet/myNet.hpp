#ifndef __MYNET_HPP__
#define __MYNET_HPP__
#include "myLayer.hpp"
#include "myBlob.hpp"
#include "RemNet.snapshotModel.pb.h"
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>

using std::unordered_map;
using std::shared_ptr;
using std::vector;
using std::string;

struct NetParam { // Whole Model parameters
	// Learning rate
	double lr;

	// Whether to update the learning rate?
	bool update_lr;

	// Learning rate decay coefficient
	double lr_decay;

	// Optimizer sgd/momentum/rmsprop/adagrad
	string optimizer;

	// Momentum optimizer coefficient decay
	double momentum;

	// Rmsprop optimizer coefficient decay
	double rmsprop;

	// L2 Regularization coefficient
	double reg;

	int epochs;

	// Whether to use mini-batch gradient descent algorithm for optimization?
	bool use_batch;

	int batch_size;

	// every acc_frequence do evaluate
	int acc_frequence;

	// Whether you need to save the model?
	bool snap_shot;

	// Save the model every few iterations
	int snapshot_interval;

	// Whether to train with fine tune
	bool fine_tune;

	// The path of the pretrained model
	string preTrainedModel;

	// layers name
	vector<string> layers;

	// layers type
	vector<string> ltypes;

	unordered_map<string, Param> lparams;

	void readNetParam(string file);
};

class Net {

public:
	void initNet(NetParam& param, vector<shared_ptr<Blob>>& x, vector<shared_ptr<Blob>>& y);
	void trainNet(NetParam& param);
	void train_with_batch(shared_ptr<Blob>& x, shared_ptr<Blob>& y, NetParam& param, string mode="TRAIN");
	void optimizer_with_batch(NetParam& param);
	void evaluate_with_batch(NetParam& param);
	void regular_with_batch(NetParam& param, string mode="TRAIN");
	double calc_accuracy(Blob& y, Blob& pred);
	void saveModelParam(shared_ptr<RemNet::snapshotModel>& snapshot_model);
	void loadModelParam(const shared_ptr<RemNet::snapshotModel>& snapshot_model);
private:
	// Train Data
	shared_ptr<Blob> x_train;
	shared_ptr<Blob> y_train;

	// Val Data
	shared_ptr<Blob> x_val;
	shared_ptr<Blob> y_val;

	vector<string> layers; // layer name
	vector<string> ltypes; // layer type

	double train_loss;
	double val_loss;
	double train_accu;
	double val_accu;

	unordered_map<string, vector<shared_ptr<Blob>>> data; // the needed Blob for forward
	
	// gradient[0]=dx, gradient[1]=dw, gradient[2]=db
	unordered_map<string, vector<shared_ptr<Blob>>> gradient; // the needed Blob for backward

	unordered_map<string, shared_ptr<Layer>> myLayers;

	unordered_map<string, vector<int>> outShapes; // output shape for each layer
	unordered_map<string, vector<shared_ptr<Blob>>> step_cache; // Preserved cumulative gradient£¬Only rmsprop and momentum are used

};

#endif