#include <iostream>
#include <cmath>
#include "TransitionMatrix.h"

bool IsPowerOfTwo(int N) {
	std::cout << N << "\n";
	std::cout << (N != 0 && (N & (N - 1)) == 0) << "\n";
	return N != 0 && (N & (N - 1)) == 0;
}

MKL_Complex16* TransitionMatrix::GetTransitionMatrix(int N) {
	if (!IsPowerOfTwo) {
		std::cout << "N is not a power of two\n";
		return nullptr;
	}
	
	int NumberOfU = static_cast<int>(std::log2(N));
	std::cout << NumberOfU;

	return nullptr;
}