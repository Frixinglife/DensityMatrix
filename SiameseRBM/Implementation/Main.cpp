#include "Experiments.h"
#include <iostream>

int main() {
    int N_v, N_h, N_a;
    N_v = N_h = N_a = 1024;
    //Experiments::GetRoMatrixAndEig(N_v, N_h, N_a);
    Experiments::GetWorkTime(N_v, N_h, N_a, true);

    //int N;
    //std::cin >> N;
    //int NumberOfSamples = 15;
    //Experiments::GetTransitionMatrixAndNewRo(N, true);
    //Experiments::GetSamples(N, NumberOfSamples);

    //int NumberOfU;
    //std::cin >> NumberOfU;
    //Experiments::CheckUnitaryMatrices(NumberOfU);

    return 0;
}