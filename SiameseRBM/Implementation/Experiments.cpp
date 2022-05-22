#include "MatrixAndVectorOperations.h"
#include "NeuralDensityOperators.h"
#include "TransitionMatrix.h"
#include "Experiments.h"
#include <iostream>
#include <iomanip>
#include <fstream>

using std::cout;

void Experiments::PrintRoMatrix(int N, MKL_Complex16* Matrix) {
    cout << "Ro matrix:" << "\n\n";

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(30) << Matrix[j + i * N];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void Experiments::PrintEigRoMatrix(int N, double* Vec) {
    cout << "Eigenvalues:\n\n";

    for (int i = 0; i < N; i++) {
        std::cout << Vec[i] << "\n";
    }
}

void Experiments::GetRoMatrixAndEig(int N_v, int N_h, int N_a, bool plot) {
    NeuralDensityOperators DensityOperators(N_v, N_h, N_a);

    DensityOperators.PrintRBMs();

    MKL_Complex16* RoMatrix = DensityOperators.GetRoMatrix(nullptr, plot);
    PrintRoMatrix(N_v, RoMatrix);

    double* EigVector = new double[N_v];

    MatrixAndVectorOperations::FindEigMatrix(N_v, RoMatrix, EigVector);
    PrintEigRoMatrix(N_v, EigVector);

    delete[]EigVector;
    delete[]RoMatrix;
}

void Experiments::GetWorkTime(int N_v, int N_h, int N_a, bool plot) {
    std::ofstream fout("times.txt", std::ios_base::app);

    NeuralDensityOperators DensityOperators(N_v, N_h, N_a);

    double work_time = 0.0;
    MKL_Complex16* RoMatrix = DensityOperators.GetRoMatrix(&work_time, plot);

    std::cout << "N_v = " << N_v << ", N_h = " << N_h << ", N_a = " << N_a << "\n";
    std::cout << "Matrix size: " << N_v << "\n";
    std::cout << "Data type: " << TYPE_OUT << "\n";
    std::cout << "Time: " << work_time << " s\n\n";

    fout << "N_v = " << N_v << ", N_h = " << N_h << ", N_a = " << N_a << "\n";
    fout << "Matrix size: " << N_v << "\n";
    fout << "Data type: " << TYPE_OUT << "\n";
    fout << "Time: " << work_time << " s\n\n";

    delete[]RoMatrix;
}

void Experiments::GetTransitionMatrixAndNewRo(int N, bool show) {
    NeuralDensityOperators DensityOperators(N, N, N);
    DensityOperators.PrintRBMs();

    MKL_Complex16* RoMatrix = DensityOperators.GetRoMatrix(nullptr);
    PrintRoMatrix(N, RoMatrix);

    TransitionMatrix TM;

    MKL_Complex16* Ub = TM.GetTransitionMatrix(N, show);
    TransitionMatrix::PrintMatrix(Ub, N, N, "Ub");

    MKL_Complex16* Ub_t = TransitionMatrix::GetHermitianConjugateMatrix(Ub, N);
    TransitionMatrix::PrintMatrix(Ub_t, N, N, "Ub_t");

    MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, Ub, Ub_t, N);
    TransitionMatrix::PrintMatrix(NewRoMatrix, N, N, "New ro matrix");

    double* NewRoMatrixDiag = new double[N];
    for (int i = 0; i < N; i++) {
        NewRoMatrixDiag[i] = NewRoMatrix[i + i * N].real();
    }

    double trace = 0.0;
    for (int i = 0; i < N; i++) {
        trace += NewRoMatrixDiag[i];
    }

    std::cout << "Diag new ro matrix:\n";
    for (int i = 0; i < N; i++) {
        std::cout << NewRoMatrixDiag[i] << "\n";
    }
    std::cout << "\n";

    std::cout << "Trace: " << trace << "\n\n";

    for (int i = 0; i < N; i++) {
        NewRoMatrixDiag[i] /= trace;
    }

    std::cout << "Diag new ro matrix after normalization:\n";
    for (int i = 0; i < N; i++) {
        std::cout << NewRoMatrixDiag[i] << "\n";
    }
    std::cout << "\n";

    delete[]RoMatrix;
    delete[]NewRoMatrix;
    delete[]NewRoMatrixDiag;
    delete[]Ub;
    delete[]Ub_t;
}

void Experiments::GetSamples(int N, int NumberOfSamples) {
    NeuralDensityOperators DensityOperators(N, N, N);
    TransitionMatrix TM;
    MKL_Complex16* RoMatrix = DensityOperators.GetRoMatrix(nullptr);
    MKL_Complex16* Ub = TM.GetTransitionMatrix(N);
    MKL_Complex16* Ub_t = TransitionMatrix::GetHermitianConjugateMatrix(Ub, N);
    MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, Ub, Ub_t, N);

    double* NewRoMatrixDiag = new double[N];
    for (int i = 0; i < N; i++) {
        NewRoMatrixDiag[i] = NewRoMatrix[i + i * N].real();
    }

    double trace = 0.0;
    for (int i = 0; i < N; i++) {
        trace += NewRoMatrixDiag[i];
    }

    for (int i = 0; i < N; i++) {
        NewRoMatrixDiag[i] /= trace;
    }

    std::cout << "Diag new ro matrix after normalization:\n";
    for (int i = 0; i < N; i++) {
        std::cout << NewRoMatrixDiag[i] << "\n";
    }
    std::cout << "\n";

    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 42);
    acc_number* random_numbers = new acc_number[NumberOfSamples];
    TRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, NumberOfSamples, random_numbers, 0.0, 1.0);
    
    int* Samples = new int[NumberOfSamples];
    for (int j = 0; j < NumberOfSamples; j++) {
        Samples[j] = 0;
        double prob_sum = 0.0;
        for (int i = 0; i < N; i++) {
            prob_sum += NewRoMatrixDiag[i];
            if (random_numbers[j] < prob_sum) {
                Samples[j] = i;
                break;
            }
        }
    }

    std::cout << "Samples:\n";
    for (int i = 0; i < NumberOfSamples; i++) {
        std::cout << Samples[i] << "\n";
    }

    delete[]Samples;
    delete[]RoMatrix;
    delete[]NewRoMatrix;
    delete[]NewRoMatrixDiag;
    delete[]Ub;
    delete[]Ub_t;
}
