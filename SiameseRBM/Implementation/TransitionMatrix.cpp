#include <iostream>
#include <iomanip>
#include <cmath>
#include "TransitionMatrix.h"

TransitionMatrix::TransitionMatrix(int seed) {
	vslNewStream(&stream, VSL_BRNG_MT19937, seed);
}

TransitionMatrix::~TransitionMatrix() {
	vslDeleteStream(&stream);
}

void TransitionMatrix::PrintMatrix(MKL_Complex16* Matrix, int n, int m, std::string name) {
	std::cout << name << ":\n";
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			std::cout << std::setw(30) << Matrix[j + i * m];
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

bool IsPowerOfTwo(int N) {
	return N != 0 && (N & (N - 1)) == 0;
}

void TransitionMatrix::GetUnitaryMatrix(MKL_Complex16* Matrix, acc_number a, acc_number b, acc_number c, acc_number d) {
	acc_number half = (acc_number)0.5;
	acc_number two = (acc_number)2.0;
	acc_number four = (acc_number)4.0;
	acc_number sqrt_number = std::sqrt(a * a - two * a * d + four * b * b + four * c * c + d * d);
	MKL_Complex16 i = MKL_Complex16(0.0, 1.0);

	MKL_Complex16 exp_plus = std::exp(half * i * (sqrt_number + a + d));
	MKL_Complex16 exp_minus = std::exp(half * i * (-sqrt_number + a + d));
	
	MKL_Complex16 elem_1 = ((sqrt_number + a - d) * exp_plus - (-sqrt_number + a - d) * exp_minus) / (two * sqrt_number);
	MKL_Complex16 elem_4 = ((sqrt_number - a + d) * exp_plus - (-sqrt_number - a + d) * exp_minus) / (two * sqrt_number);

	MKL_Complex16 elem_2 = ((b - i * c) * exp_plus - (b - i * c) * exp_minus) / sqrt_number;
	MKL_Complex16 elem_3 = ((b + i * c) * exp_plus - (b + i * c) * exp_minus) / sqrt_number;

	Matrix[0] = elem_1;
	Matrix[1] = elem_2;
	Matrix[2] = elem_3;
	Matrix[3] = elem_4;
}

void TransitionMatrix::GetUnitaryMatrices(MKL_Complex16* Matrices, int NumberOfU, acc_number left, acc_number right) {
	int N = 4 * NumberOfU;
	acc_number* ElementsU = new acc_number[N];
	TRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, ElementsU, left, right);
	
	for (int index = 0; index < N; index += 4) {
		acc_number a = ElementsU[index];
		acc_number b = ElementsU[index + 1];
		acc_number c = ElementsU[index + 2];
		acc_number d = ElementsU[index + 3];
		GetUnitaryMatrix(Matrices + index, a, b, c, d);
	}

	delete[]ElementsU;
}

void TransitionMatrix::ShowUnitaryMatrices(MKL_Complex16* Matrices, int NumberOfU) {
	std::cout << "Number of U: " << NumberOfU << "\n";
	std::cout << "U matrices:\n";
	for (int i = 0; i < 2 * NumberOfU; i++) {
		for (int j = 0; j < 2; j++) {
			std::cout << std::setw(30) << Matrices[j + i * 2];
		}
		std::cout << ((i % 2 == 1) ? "\n\n" : "\n");
	}
	std::cout << "\n";
}

MKL_Complex16 TransitionMatrix::ComplexMult(MKL_Complex16 A, MKL_Complex16 B) {
	return MKL_Complex16(A.real() * B.real() - A.imag() * B.imag(), A.real() * B.imag() + B.real() * A.imag());
}

void TransitionMatrix::KroneckerProduction(MKL_Complex16* Matrix_A, int size_A, MKL_Complex16* Matrix_B, int size_B, MKL_Complex16* Matrix_Res, bool show) {
	if (show) {
		TransitionMatrix::PrintMatrix(Matrix_A, size_A, size_A, "A");
		TransitionMatrix::PrintMatrix(Matrix_B, size_B, size_B, "B");
	}

	int index = 0;
	for (int i = 0; i < size_A; i++) {
		for (int k = 0; k < size_B; k++) {
			for (int j = 0; j < size_A; j++) {
				for (int l = 0; l < size_B; l++) {
					Matrix_Res[index++] = ComplexMult(Matrix_A[j + i * size_A], Matrix_B[l + k * size_B]);
				}
			}
		}
	}

	if (show) {
		TransitionMatrix::PrintMatrix(Matrix_Res, size_A * size_B, size_A * size_B, "A (x) B");
	}
}

MKL_Complex16* TransitionMatrix::GetTransitionMatrix(int N, bool show) {
	bool PowerOfTwo = IsPowerOfTwo(N);
	if (!PowerOfTwo) {
		std::cout << "N is not a power of two\n";
		return nullptr;
	}
	
	int NumberOfU = static_cast<int>(std::log2(N));
	
	MKL_Complex16* MatricesU = new MKL_Complex16[4 * NumberOfU];
	GetUnitaryMatrices(MatricesU, NumberOfU);

	if (show) {
		ShowUnitaryMatrices(MatricesU, NumberOfU);
	}

	int start_size = 2;
	int bias = 4;
	MKL_Complex16* TransitionMatrix = new MKL_Complex16[start_size * start_size];
	for (int i = 0; i < 4; i++) {
		TransitionMatrix[i] = MatricesU[i];
	}

	for (int i = 0; i < NumberOfU - 1; i++) {
		MKL_Complex16* Matrix_Res = new MKL_Complex16[start_size * start_size * 4];

		KroneckerProduction(TransitionMatrix, start_size, MatricesU + bias, 2, Matrix_Res, show);

		bias += 4;
		start_size *= 2;
		delete[]TransitionMatrix;

		TransitionMatrix = new MKL_Complex16[start_size * start_size];
		for (int j = 0; j < start_size * start_size; j++) {
			TransitionMatrix[j] = Matrix_Res[j];
		}

		delete[]Matrix_Res;
	}

	return TransitionMatrix;
}

MKL_Complex16* TransitionMatrix::GetHermitianConjugateMatrix(MKL_Complex16* Matrix, int N) {
	MKL_Complex16* Result = new MKL_Complex16[N * N];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			MKL_Complex16 number = Matrix[i + j * N];
			Result[j + i * N] = MKL_Complex16(number.real(), -number.imag());
		}
	}

	return Result;
}

MKL_Complex16* TransitionMatrix::GetNewRoMatrix(MKL_Complex16* Ro, MKL_Complex16* Ub, MKL_Complex16* Ub_t, int N) {
	MKL_Complex16* Temp = new MKL_Complex16[N * N];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				Temp[j + i * N] += ComplexMult(Ub[k + i * N], Ro[j + k * N]);
			}	
		}
	}

	MKL_Complex16* NewRo = new MKL_Complex16[N * N];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				NewRo[j + i * N] += ComplexMult(Temp[k + i * N], Ub_t[j + k * N]);
			}
		}
	}

	delete[]Temp;

	return NewRo;
}