#ifndef _TRANSITION_MATRIX_H_
#define _TRANSITION_MATRIX_H_

#include "ComplexMKL.h"
#include "DataType.h"

class TransitionMatrix {
public:
    static MKL_Complex16* GetTransitionMatrix(int N);
};

#endif //_TRANSITION_MATRIX_H_
