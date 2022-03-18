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

    double GetGamma(int N, acc_number* FirstSigma, acc_number* SecondSigma, char PlusOrMinus);
    MKL_Complex16 GetPi(int N, acc_number* FirstSigma, acc_number* SecondSigma);
    MKL_Complex16 GetRo(int N, acc_number* FirstSigma, acc_number* SecondSigma);
    MKL_Complex16* GetRoMatrix(double *work_time = nullptr, bool plot = false);
};

#endif //_NEURAL_DENSITY_OPERATORS_H_
