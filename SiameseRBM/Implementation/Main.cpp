#include "Experiments.h"
#include <iostream>

int main() {
    //int N_v, N_h, N_a;
    //N_v = N_h = N_a = 8;
    //Experiments::GetRoMatrixAndEig(N_v, N_h, N_a);
    //Experiments::GetWorkTime(N_v, N_h, N_a, true);

    int N;
    std::cin >> N;
    int NumberOfSamples = 10;
    Experiments::GetTransitionMatrixAndNewRo(N, true);
    Experiments::GetSamples(N, NumberOfSamples);

    return 0;
}