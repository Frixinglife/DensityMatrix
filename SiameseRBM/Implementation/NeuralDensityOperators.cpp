#include "MatrixAndVectorOperations.h"
#include "NeuralDensityOperators.h"
#include "RandomMatricesForRBM.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include "omp.h"

using std::cout;

acc_number ONE = (acc_number)1.0;
acc_number HALF = (acc_number)0.5;
acc_number ZERO = (acc_number)0.0;

NeuralDensityOperators::NeuralDensityOperators(int N_v, int N_h, int N_a) {
    const int N_h_N_v = N_h * N_v;
    const int N_a_N_v = N_a * N_v;

    acc_number* W_1 = new acc_number[N_h_N_v];
    acc_number* V_1 = new acc_number[N_a_N_v];
    acc_number* b_1 = new acc_number[N_v];
    acc_number* c_1 = new acc_number[N_h];
    acc_number* d_1 = new acc_number[N_a];

    acc_number* W_2 = new acc_number[N_h_N_v];
    acc_number* V_2 = new acc_number[N_a_N_v];
    acc_number* b_2 = new acc_number[N_v];
    acc_number* c_2 = new acc_number[N_h];
    acc_number* d_2 = new acc_number[N_a];

    RandomMatricesForRBM Random(42);

    Random.GetRandomMatrix(W_1, N_h, N_v);
    Random.GetRandomMatrix(W_2, N_h, N_v);

    Random.GetRandomMatrix(V_1, N_a, N_v);
    Random.GetRandomMatrix(V_2, N_a, N_v);

    Random.GetRandomVector(b_1, N_v);
    Random.GetRandomVector(b_2, N_v);

    Random.GetRandomVector(c_1, N_h);
    Random.GetRandomVector(c_2, N_h);

    Random.GetRandomVector(d_1, N_a);
    Random.GetRandomVector(d_2, N_a);

    FirstSiameseRBM.SetSiameseRBM(N_v, N_h, N_a, W_1, V_1, b_1, c_1, d_1);
    SecondSiameseRBM.SetSiameseRBM(N_v, N_h, N_a, W_2, V_2, b_2, c_2, d_2);

    delete[]W_1;
    delete[]W_2;
    delete[]V_1;
    delete[]V_2;
    delete[]b_1;
    delete[]b_2;
    delete[]c_1;
    delete[]c_2;
    delete[]d_1;
    delete[]d_2;
}

void NeuralDensityOperators::PrintRBMs() const {
    FirstSiameseRBM.PrintSiameseRBM("First siamese RBM");
    SecondSiameseRBM.PrintSiameseRBM("Second siamese RBM");
}

double NeuralDensityOperators::GetGamma(int N, acc_number* FirstSigma, acc_number* SecondSigma, char PlusOrMinus) {
    int N_h, N_v;
    double Answer = 0.0;
    acc_number* FirstVec, * SecondVec, * IntermedVec;

    N_h = FirstSiameseRBM.N_h;
    N_v = FirstSiameseRBM.N_v;

    IntermedVec = new acc_number[N];
    FirstVec = new acc_number[N_h];
    SecondVec = new acc_number[N_h];

    switch (PlusOrMinus) {
    case '+':
        MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, IntermedVec);
        Answer = (double)MatrixAndVectorOperations::ScalarVectorMult(N, FirstSiameseRBM.b, IntermedVec);

        MatrixAndVectorOperations::MKL_MatrixVectorMult(N_h, N_v, FirstSiameseRBM.W, FirstSigma, FirstVec);
        MatrixAndVectorOperations::MKL_MatrixVectorMult(N_h, N_v, FirstSiameseRBM.W, SecondSigma, SecondVec);

        MatrixAndVectorOperations::VectorsAdd(N_h, FirstVec, FirstSiameseRBM.c, FirstVec);
        MatrixAndVectorOperations::VectorsAdd(N_h, SecondVec, FirstSiameseRBM.c, SecondVec);
        
        for (int i = 0; i < N_h; ++i) {
            double First = (double)FirstVec[i];
            double Second = (double)SecondVec[i];
            Answer += std::log(1.0 + std::exp(First)) + std::log(1.0 + std::exp(Second));
        }

        break;

    case '-':
        MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, IntermedVec);
        Answer = (double)MatrixAndVectorOperations::ScalarVectorMult(N, SecondSiameseRBM.b, IntermedVec);

        MatrixAndVectorOperations::MKL_MatrixVectorMult(N_h, N_v, SecondSiameseRBM.W, FirstSigma, FirstVec);
        MatrixAndVectorOperations::MKL_MatrixVectorMult(N_h, N_v, SecondSiameseRBM.W, SecondSigma, SecondVec);

        MatrixAndVectorOperations::VectorsAdd(N_h, FirstVec, SecondSiameseRBM.c, FirstVec);
        MatrixAndVectorOperations::VectorsAdd(N_h, SecondVec, SecondSiameseRBM.c, SecondVec);

        for (int i = 0; i < N_h; ++i) {
            double First = (double)FirstVec[i];
            double Second = (double)SecondVec[i];
            Answer += std::log(1.0 + std::exp(First)) - std::log(1.0 + std::exp(Second));
        }

        break;

    default:
        break;
    }

    delete[]IntermedVec;
    delete[]FirstVec;
    delete[]SecondVec;

    Answer *= 0.5;

    return Answer;
}

MKL_Complex16 NeuralDensityOperators::GetPi(int N, acc_number* FirstSigma, acc_number* SecondSigma) {
    MKL_Complex16 Answer(0.0, 0.0);
    MKL_Complex16 One(1.0, 0.0);
    int N_a = FirstSiameseRBM.N_a;
    int N_v = FirstSiameseRBM.N_v;
    acc_number* FirstVec, * SecondVec, * Vec;
    
    Vec = new acc_number[N];
    FirstVec = new acc_number[N_a];
    SecondVec = new acc_number[N_a];

    MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, Vec);
    MatrixAndVectorOperations::MKL_MatrixVectorMult(N_a, N_v, FirstSiameseRBM.V, Vec, FirstVec);

    MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, Vec);
    MatrixAndVectorOperations::MKL_MatrixVectorMult(N_a, N_v, SecondSiameseRBM.V, Vec, SecondVec);

    MatrixAndVectorOperations::MultVectorByNumber(N_a, FirstVec, HALF, FirstVec);
    MatrixAndVectorOperations::MultVectorByNumber(N_a, SecondVec, HALF, SecondVec);

    MatrixAndVectorOperations::VectorsAdd(N_a, FirstVec, FirstSiameseRBM.d, FirstVec);    

    for (int i = 0; i < N_a; ++i) {
        MKL_Complex16 CurrentAnswer((double)FirstVec[i], (double)SecondVec[i]);
        Answer += std::log(One + std::exp(CurrentAnswer));
    }

    delete[]Vec;
    delete[]FirstVec;
    delete[]SecondVec;

    return Answer;
}

MKL_Complex16 NeuralDensityOperators::GetRo(int N, acc_number* FirstSigma, acc_number* SecondSigma) {
    MKL_Complex16 Gamma(GetGamma(N, FirstSigma, SecondSigma, '+'), GetGamma(N, FirstSigma, SecondSigma, '-'));
    MKL_Complex16 Pi = GetPi(N, FirstSigma, SecondSigma);

    return std::exp(Gamma + Pi);
}

MKL_Complex16* NeuralDensityOperators::GetRoMatrix(double *work_time, bool plot) {
    int N_v = FirstSiameseRBM.N_v;
    const int N_v_N_v = N_v * N_v;
    MKL_Complex16* RoMatrix = new MKL_Complex16[N_v_N_v];

    acc_number* FirstSigma, * SecondSigma;
    FirstSigma = new acc_number[N_v];
    SecondSigma = new acc_number[N_v];

    for (int i = 0; i < N_v; i++) {
        FirstSigma[i] = 0.0;
        SecondSigma[i] = 0.0;
    }

    unsigned int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    auto start = std::chrono::high_resolution_clock::now();

    //#pragma omp parallel for
    for (int i = 0; i < N_v; i++) {
        for (int j = 0; j < N_v; j++) {
            FirstSigma[i] = ONE;
            SecondSigma[j] = ONE;

            RoMatrix[j + i * N_v] = GetRo(N_v, FirstSigma, SecondSigma);

            FirstSigma[i] = ZERO;
            SecondSigma[j] = ZERO;
        }
    }

    MKL_Complex16 Sum(0.0, 0.0);

    for (int i = 0; i < N_v; i++) {
        Sum += RoMatrix[i + i * N_v];
    }

    for (int i = 0; i < N_v; i++) {
        for (int j = 0; j < N_v; j++) {
            RoMatrix[j + i * N_v] /= Sum;
        }
    }

    auto diff = std::chrono::high_resolution_clock::now() - start;

    delete[]FirstSigma;
    delete[]SecondSigma;

    if (work_time != nullptr) {
        *work_time = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(diff).count());
    }   

    if (plot) {
        std::ofstream fout(MATRIX_OUT, std::ios_base::out | std::ios_base::trunc);

        fout << N_v << "\n";

        for (int i = 0; i < N_v; i++) {
            fout << RoMatrix[i + i * N_v].real() << "\n";
        }
    }

    return RoMatrix;
}
