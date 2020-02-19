#include <iostream>
#include <string>
#include <memory>

#include "myBlob.hpp"
#include "myNet.hpp"
using namespace std;

int ReverseInt(int i) { // Data conversion
	unsigned char ch1, ch2, ch3, ch4;  // one int = four char
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMnistData(string path, shared_ptr<Blob>& images) {
	ifstream file(path, ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		cout << "number_of_images = " << number_of_images << endl;
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		cout << "n_rows = " << n_rows << endl;
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		cout << "n_cols = " << n_cols << endl;

		for (int i = 0; i < number_of_images; i++) {
			for (int h = 0; h < n_rows; h++) {
				for (int w = 0; w < n_cols; w++) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));		
					(*images)[i](h, w, 0) = (double)temp / 255;
				}
			}
		}
	}
	else
		cout << "no data file found :-(" << endl;

}
void ReadMnistLabel(string path, shared_ptr<Blob>& labels) {
	ifstream file(path, ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		cout << "number_of_Labels = " << number_of_images << endl;
		for (int i = 0; i < number_of_images; ++i) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			(*labels)[i](0, 0, (int)temp) = 1;
		}
	}
	else
		cout << "no label file found :-(" << endl;
}

void trainModel(NetParam& net_param, shared_ptr<Blob> x_train_ori, shared_ptr<Blob> y_train_ori) {

	vector<string> layers = net_param.layers;
	vector<string> ltypes = net_param.ltypes;


	// 1. The 60,000 pictures were divided into training set and test set at a ratio of 59:1
	shared_ptr<Blob> x_train(new Blob(x_train_ori->subBlob(0, 59000)));
	shared_ptr<Blob> y_train(new Blob(y_train_ori->subBlob(0, 59000)));

	shared_ptr<Blob> x_val(new Blob(x_train_ori->subBlob(59000, 60000)));
	shared_ptr<Blob> y_val(new Blob(y_train_ori->subBlob(59000, 60000)));

	vector<shared_ptr<Blob>> xx{ x_train, x_val };
	vector<shared_ptr<Blob>> yy{ y_train, y_val };

	// 2. Initializes the network structure
	Net myModel;
	myModel.initNet(net_param, xx, yy);

	// 3. Train start
	cout << "----------------Train start...----------------" << endl;

	myModel.trainNet(net_param);

	cout << "----------------Train end...----------------" << endl;
}

void trainModel_with_exVal(NetParam& net_param, shared_ptr<Blob> x_train_ori, shared_ptr<Blob> y_trian_ori, 
								shared_ptr<Blob> x_val_ori, shared_ptr<Blob> y_val_ori) {
	vector<string> layers = net_param.layers;
	vector<string> ltypes = net_param.ltypes;

	vector<shared_ptr<Blob>> xx{ x_train_ori, x_val_ori };
	vector<shared_ptr<Blob>> yy{ y_trian_ori, y_val_ori };

	// Initializes the network structure
	Net myModel;
	myModel.initNet(net_param, xx, yy);

	// Train start
	cout << "----------------Train start...----------------" << endl;

	myModel.trainNet(net_param);

	cout << "----------------Train end...----------------" << endl;
}

int main(int argc, char** argv) {
	string configFile = "./myModel.json";
	NetParam net_param;

	// 0. Read myModel.json, and parse
	net_param.readNetParam(configFile);

	// create two Blob object£¬one save images£¬one save labels
	shared_ptr<Blob> images_train(new Blob(60000, 1, 28, 28, TZEROS));
	shared_ptr<Blob> labels_train(new Blob(60000, 10, 1, 1, TZEROS)); // one-hot
	ReadMnistData("mnist_data/train/train-images.idx3-ubyte", images_train);
	ReadMnistLabel("mnist_data/train/train-labels.idx1-ubyte", labels_train);

	shared_ptr<Blob> images_test(new Blob(10000, 1, 28, 28, TZEROS));
	shared_ptr<Blob> labels_test(new Blob(10000, 10, 1, 1, TZEROS)); // one-hot
	ReadMnistData("mnist_data/test/t10k-images.idx3-ubyte", images_test);
	ReadMnistLabel("mnist_data/test/t10k-labels.idx1-ubyte", labels_test);

	int samples_num = 1000;
	shared_ptr<Blob> x_train(new Blob(images_train->subBlob(0, samples_num)));
	shared_ptr<Blob> y_train(new Blob(labels_train->subBlob(0, samples_num)));
	shared_ptr<Blob> x_test(new Blob(images_test->subBlob(0, samples_num)));
	shared_ptr<Blob> y_test(new Blob(labels_test->subBlob(0, samples_num)));
	
	trainModel_with_exVal(net_param, x_train, y_train, x_test, y_test);

}