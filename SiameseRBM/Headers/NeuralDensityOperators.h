#ifndef _NEURAL_DENSITY_OPERATORS_H_
#define _NEURAL_DENSITY_OPERATORS_H_

#include "DataType.h"
#include "ComplexMKL.h"
#include "SiameseRBM.h"

class NeuralDensityOperators {
public:
    SiameseRBM FirstSiameseRBM, SecondSiameseRBM;

    NeuralDensityOperators(int N_v, int N_h, int N_a);
    ~NeuralDensityOperators() {};

    void PrintRBMs() const;

    acc_number GetGamma(int N, acc_number* FirstSigma, acc_number* SecondSigma, char PlusOrMinus);
    TComplex GetPi(int N, acc_number* FirstSigma, acc_number* SecondSigma);
    TComplex GetRo(int N, acc_number* FirstSigma, acc_number* SecondSigma);
    TComplex* GetRoMatrix(double *work_time = nullptr, bool plot = false);
};

#endif //_NEURAL_DENSITY_OPERATORS_H_
