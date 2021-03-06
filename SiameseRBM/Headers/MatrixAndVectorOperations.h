#ifndef _MATRIX_AND_VECTOR_OPERATIONS_H_
#define _MATRIX_AND_VECTOR_OPERATIONS_H_

#include "ComplexMKL.h"
#include "DataType.h"
#include <string>

class MatrixAndVectorOperations {
public:
    static void VectorsAdd(int N, acc_number* FirstVec, acc_number* SecondVec, acc_number* Result);
    static void VectorsSub(int N, acc_number* FirstVec, acc_number* SecondVec, acc_number* Result);
    static acc_number ScalarVectorMult(int N, acc_number* FirstVec, acc_number* SecondVec);
    static void MultVectorByNumber(int N, acc_number* Vec, acc_number Number, acc_number* Result);
    static void MatrixVectorMult(int N, int M, acc_number* Matrix, acc_number* Vec, acc_number* Result);
    static void MKL_MatrixVectorMult(int N, int M, acc_number* Matrix, acc_number* Vec, acc_number* Result);
    static void PrintVector(int N, acc_number* Vec, const std::string& Text);
    static void FindEigMatrix(int N, MKL_Complex16* Matrix, double* Result);
};

#endif //_MATRIX_AND_VECTOR_OPERATIONS_H_
