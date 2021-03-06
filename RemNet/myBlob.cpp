#include "myBlob.hpp"
#include "cassert"
using namespace std;
using namespace arma;

Blob::Blob(const int n, const int c, const int h, const int w, int type) :N(n), C(c), H(h), W(w) {
	arma_rng::set_seed_random(); //	System randomly generates seeds
	init(N, C, H, W, type);
}

void Blob::init(const int n, const int c, const int h, const int w, int type) {
	if (type == TZEROS) {
		blob_data = vector<cube>(n, cube(h, w, c, fill::zeros)); // Manage n cubes with vector
		return;
	}
	if (type == TONES) {
		blob_data = vector<cube>(n, cube(h, w, c, fill::ones));
		return;
	}
	if (type == TDEFAULT) {
		blob_data = vector<cube>(n, cube(h, w, c));
		return;
	}
	if (type == TRANDU) {
		for (int i = 0; i < n; i++) // generate n cubes
			blob_data.push_back(arma::randu<cube>(h, w, c)); // Uniform distribution
		return;
	}

	if (type == TRANDN) {
		for (int i = 0; i < n; i++)
			blob_data.push_back(arma::randn<cube>(h, w, c)); // Standard normal distribution
		return;
	}
}

void Blob::print(string str) {
	assert(!blob_data.empty());
	cout << str << endl;
	for (int i = 0; i < N; i++) { // N is the number of cubes in the blob
		printf("N = %d\n", i);
		this->blob_data[i].print(); // Call cube's own print method
	}
}

cube& Blob::operator[] (int i) {
	return blob_data[i];
}

vector<cube>& Blob::get_data() {
	return blob_data;
}

Blob Blob::subBlob(int start, int end) {
	if (end > start) {
		Blob tmp(end - start, C, H, W);
		for (int i = start; i < end; i++)
			tmp[i - start] = (*this)[i];
		return tmp;
	} else {
		Blob tmp(N-start+end, C, H, W);
		for (int i = start; i < N; i++)
			tmp[i - start] = (*this)[i];
		for (int i = 0; i < end; i++)
			tmp[i + N - start] = (*this)[i];
		return tmp;
	}
}

Blob& Blob::operator*=(double k) {
	for (int i = 0; i < N; i++)
		blob_data[i] = blob_data[i] * k;
	return *this;
}

Blob Blob::pad(int pad, double val) {
	assert(!blob_data.empty());
	Blob padX(N, C, H + (pad << 1), W + (pad << 1));
	padX = val;

	for (int n = 0; n < N; n++) 
		for (int c = 0; c < C; c++) 
			for (int h = 0; h < H; h++) 
				for (int w = 0; w < W; w++) 
					padX[n](h + pad, w + pad, c) = blob_data[n](h, w, c);
	return padX;
}
void Blob::maxIn(double val) {
	assert(!blob_data.empty());
	for (int i = 0; i < N; i++) {
		blob_data[i].transform([val](double e) {return e > val ? e : val; });
		// clipping
		double clipped = 6.0;
		blob_data[i].transform([clipped](double e) {return e > clipped ? clipped : e; });
	}
	return;
}

void Blob::convertIn(double val) {
	assert(!blob_data.empty());
	for (int i = 0; i < N; i++)
		blob_data[i].transform([val](double e) {return e > val ? 0 : 1; });
	return;
}

vector<int> Blob::size() const {
	vector<int> shape{N, C, H, W};
	return shape;
}

Blob::Blob(const vector<int> shape, int type) : N(shape[0]), C(shape[1]), H(shape[2]), W(shape[3]) {
	arma_rng::set_seed_random();
	init(N, C, H, W, type);
}

Blob operator*(Blob A, Blob B) {
	// Make sure both input blobs are the same size
	vector<int> size_A = A.size();
	vector<int> size_B = B.size();
	for (int i = 0; i < 4; i++)
		assert(size_A[i] == size_B[i]);
	Blob C(A.size());
	int N = size_A[0];
	// Traverse all cubes and multiply the corresponding positions of each cube (cube % cube)
	for (int i = 0; i < N; i++)
		C[i] = A[i] % B[i];
	return C;
}

Blob Blob::unPad(int pad) {
	assert(!blob_data.empty());
	Blob out(N, C, H - (pad << 1), W - (pad << 1));
	for (int n = 0; n < N; n++) 
		for (int c = 0; c < C; c++)
			for (int h = pad; h < H - pad; h++) 
				for (int w = pad; w < W - pad; w++)
					out[n](h - pad, w - pad, c) = blob_data[n](h, w, c);
	return out;
}

Blob operator*(double num, Blob B) {
	// Iterate over all cubes, multiplying each cube by a value
	int N = B.getN();
	Blob out(B.size());
	for (int i = 0; i < N; i++)
		out[i] = num * B[i];
	return out;
}

Blob operator+(Blob A, Blob B) {
	// Make sure both input blobs are the same size
	vector<int> size_A = A.size();
	vector<int> size_B = B.size();
	for (int i = 0; i < 4; ++i)
		assert(size_A[i] == size_B[i]);
	int N = size_A[0];
	// Traverse all cubes and add the corresponding positions of each cube (cube + cube)
	Blob C(A.size());
	for (int i = 0; i < N; ++i)
		C[i] = A[i] + B[i];
	return C;
}

Blob operator+(Blob A, double val) {
	int N = A.getN();
	Blob out(A.size());
	for (int i = 0; i < N; ++i)
		out[i] = A[i] + val;
	return out;
}

Blob operator/(Blob A, Blob B) {
	vector<int> size_A = A.size();
	vector<int> size_B = B.size();
	for (int i = 0; i < 4; i++)
		assert(size_A[i] == size_B[i]);
	Blob C(A.size());
	// Traverse all cubes and do phase division for each cube corresponding position (cube / cube)
	int N = size_A[0];
	for (int i = 0; i < N; i++)
		C[i] = A[i] / B[i];
	return C;
}


Blob sqrt(Blob A) {
	int N = A.getN();
	Blob out(A.size());
	for (int i = 0; i < N; ++i)
		out[i] = arma::sqrt(A[i]);
	return out;
}

Blob operator/(Blob A, double val) {
	int N = A.getN();
	Blob out(A.size());
	for (int i = 0; i < N; i++)
		out[i] = A[i] / val;
	return out;
}

Blob square(Blob A) {
	int N = A.getN();
	Blob out(A.size());
	for (int i = 0; i < N; i++)
		out[i] = arma::square(A[i]);
	return out;
}

double accu(Blob A) {
	double res = 0;
	int N = A.getN();
	Blob out(A.size());
	for (int i = 0; i < N; i++)
		res += arma::accu(A[i]);
	return res;
}