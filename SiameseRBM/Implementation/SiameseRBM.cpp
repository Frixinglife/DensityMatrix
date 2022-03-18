#include "SiameseRBM.h"
#include <iostream>

using std::cout;

SiameseRBM::~SiameseRBM() {
    delete[]W;
    delete[]V;
    delete[]b;
    delete[]c;
    delete[]d;
}

void SiameseRBM::SetSiameseRBM(int _N_v, int _N_h, int _N_a, acc_number* _W,
    acc_number* _V, acc_number* _b, acc_number* _c, acc_number* _d) {

    N_v = _N_v;
    N_h = _N_h;
    N_a = _N_a;

    const int N_h_N_v = N_h * N_v;
    const int N_a_N_v = N_a * N_v;

    W = new acc_number[N_h_N_v];
    V = new acc_number[N_a_N_v];
    b = new acc_number[N_v];
    c = new acc_number[N_h];
    d = new acc_number[N_a];

    for (int i = 0; i < N_h_N_v; i++) {
        W[i] = _W[i];
    }

    for (int i = 0; i < N_a_N_v; i++) {
        V[i] = _V[i];
    }

    for (int i = 0; i < N_v; i++) {
        b[i] = _b[i];
    }

    for (int i = 0; i < N_h; i++) {
        c[i] = _c[i];
    }

    for (int i = 0; i < N_a; i++) {
        d[i] = _d[i];
    }
}

void SiameseRBM::PrintSiameseRBM(const std::string& NameRBM) const {
    cout << "\n" << NameRBM << ":\n\n";
    cout << "N_v = " << N_v << ", N_h = " << N_h << ", N_a = " << N_a << "\n\n";

    cout << "W:" << "\n";
    for (int i = 0; i < N_h; ++i) {
        for (int j = 0; j < N_v; ++j) {
            cout << W[j + i * N_v] << " ";
        }
        cout << "\n";
    }
    cout << "\n";

    cout << "V:" << "\n";
    for (int i = 0; i < N_a; ++i) {
        for (int j = 0; j < N_v; ++j) {
            cout << V[j + i * N_v] << " ";
        }
        cout << "\n";
    }
    cout << "\n";

    cout << "b:" << "\n";
    for (int i = 0; i < N_v; ++i) {
        cout << b[i] << "\n";
    }
    cout << "\n";

    cout << "c:" << "\n";
    for (int i = 0; i < N_h; ++i) {
        cout << c[i] << "\n";
    }
    cout << "\n";

    cout << "d:" << "\n";
    for (int i = 0; i < N_a; ++i) {
        cout << d[i] << "\n";
    }
    cout << "\n";
}