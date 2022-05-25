#ifndef _EXPERIMENTS_H_
#define _EXPERIMENTS_H_

#include "ComplexMKL.h"
#include "DataType.h"

class Experiments {
public:
    static void PrintRoMatrix(int N, MKL_Complex16* Matrix);
    static void PrintEigRoMatrix(int N, double* Vec);
    static void GetRoMatrixAndEig(int N_v, int N_h, int N_a, bool plot = false);
    static void GetWorkTime(int N_v, int N_h, int N_a, bool plot = false);
    static void GetTransitionMatrixAndNewRo(int N, bool show = false);
    static void GetSamples(int N, int NumberOfSamples);
    static void CheckUnitaryMatrices(int NumberOfU);
};

#endif //_EXPERIMENTS_H_
