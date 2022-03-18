#ifndef _EXPERIMENTS_H_
#define _EXPERIMENTS_H_

#include "ComplexMKL.h"
#include "DataType.h"

class Experiments {
public:
    static void PrintRoMatrix(int N, TComplex* Matrix);
    static void PrintEigRoMatrix(int N, acc_number* Vec);
    static void GetRoMatrixAndEig(int N_v, int N_h, int N_a, bool plot = false);
    static void GetWorkTime(int N_v, int N_h, int N_a, bool plot = false);
};

#endif //_EXPERIMENTS_H_
