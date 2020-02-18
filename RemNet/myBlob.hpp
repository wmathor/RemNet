#ifndef __MYBLOB_HPP__
#define __MYBLOB_HPP__
#include <vector>
#include <armadillo>

using std::vector;
using std::string;
using arma::cube;

enum FillType {
	TZEROS = 0,  // Use 0 to fill the cube
	TONES = 1,   // Use 1 to fill the cube
	TRANDU = 2,  // Fill the cube with evenly distributed values between 0 and 1
	TRANDN = 3,  // Use mu=0 and sigma=1 Gaussian distribution to fill the cube
	TDEFAULT = 4
};

class Blob {

public:
	Blob() :N(0), C(0), H(0), W(0) {};
	Blob(const int n, const int c, const int h, const int w, int type = TDEFAULT);
	Blob(const vector<int> shape, int type = TDEFAULT);
	void print(string str = "");
	cube& operator[](int i);
	Blob& operator*=(double i);
	Blob& operator=(double val);
	friend Blob operator*(Blob A, Blob B);
	friend Blob operator*(double num, Blob B);
	friend Blob operator+(Blob A, Blob B);
	friend Blob operator+(Blob A, double val);
	friend Blob operator/(Blob A, Blob B);
	friend Blob sqrt(Blob A);
	friend Blob square(Blob A);
	friend double accu(Blob A);
	friend Blob operator/(Blob A, double val);
	vector<cube>& get_data();
	Blob subBlob(int start, int end);
	Blob pad(int pad, double val = 0);
	void maxIn(double val = 0);
	void convertIn(double val = 0);
	vector<int> size() const;
	Blob unPad(int pad);
	inline int getN() const { return N; }
	inline int getC() const { return C; }
	inline int getH() const { return H; }
	inline int getW() const { return W; }

private:
	int N; // number of cube(feature map)
	int C; // channels
	int H; // height
	int W; // width
	vector<cube> blob_data;

	void init(const int n, const int c, const int h, const int w, int type);

};


#endif