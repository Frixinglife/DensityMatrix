#include "Experiments.h"
#include "TransitionMatrix.h"

int main() {
    //int N_v, N_h, N_a;
    //N_v = N_h = N_a = 8;
    //Experiments::GetRoMatrixAndEig(N_v, N_h, N_a);
    //Experiments::GetWorkTime(N_v, N_h, N_a, true);

    int N = 32;
    MKL_Complex16* U_b = TransitionMatrix::GetTransitionMatrix(N);
    delete[]U_b;

    return 0;
}
