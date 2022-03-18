#include "Experiments.h"

int main() {
    int N_v, N_h, N_a;
    N_v = N_h = N_a = 200;
    //Experiments::GetRoMatrixAndEig(N_v, N_h, N_a);
    Experiments::GetWorkTime(N_v, N_h, N_a, true);
    return 0;
}
