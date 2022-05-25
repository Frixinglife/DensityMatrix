#ifndef _TRANSITION_MATRIX_H_
#define _TRANSITION_MATRIX_H_

#include "ComplexMKL.h"
#include "DataType.h"

class TransitionMatrix {
private:
    VSLStreamStatePtr stream;

public:
    TransitionMatrix(int seed = 42);
    ~TransitionMatrix();

    MKL_Complex16* GetTransitionMatrix(int N, bool show = false);
    void GetUnitaryMatrices(MKL_Complex16* Matrices, int NumberOfU, 
        acc_number left = (acc_number)0.0, acc_number right = (acc_number)1.0);
    void ShowUnitaryMatrices(MKL_Complex16* Matrices, int NumberOfU);
    void KroneckerProduction(MKL_Complex16* Matrix_A, int size_A, MKL_Complex16* Matrix_B, int size_B,
        MKL_Complex16* Matrix_Res, bool show = false);
    static MKL_Complex16* GetHermitianConjugateMatrix(MKL_Complex16* Matrix, int N);
    static MKL_Complex16* GetNewRoMatrix(MKL_Complex16* Ro, MKL_Complex16* Ub, MKL_Complex16* Ub_t, int N);
    static void PrintMatrix(MKL_Complex16* Matrix, int n, int m, std::string name);
};

#endif //_TRANSITION_MATRIX_H_
