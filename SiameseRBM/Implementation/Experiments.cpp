#include "MatrixAndVectorOperations.h"
#include "NeuralDensityOperators.h"
#include "Experiments.h"
#include <iostream>
#include <iomanip>
#include <fstream>

using std::cout;

void Experiments::PrintRoMatrix(int N, TComplex* Matrix) {
    cout << "Ro matrix:" << "\n\n";

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(30) << Matrix[j + i * N];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void Experiments::PrintEigRoMatrix(int N, acc_number* Vec) {
    cout << "Eigenvalues:\n\n";

    for (int i = 0; i < N; i++) {
        std::cout << Vec[i] << "\n";
    }
}

void Experiments::GetRoMatrixAndEig(int N_v, int N_h, int N_a, bool plot) {
    NeuralDensityOperators DensityOperators(N_v, N_h, N_a);

    DensityOperators.PrintRBMs();

    TComplex* RoMatrix = DensityOperators.GetRoMatrix(nullptr, plot);
    PrintRoMatrix(N_v, RoMatrix);

    acc_number* EigVector = new acc_number[N_v];

    MatrixAndVectorOperations::FindEigMatrix(N_v, RoMatrix, EigVector);
    PrintEigRoMatrix(N_v, EigVector);

    delete[]EigVector;
    delete[]RoMatrix;
}

void Experiments::GetWorkTime(int N_v, int N_h, int N_a, bool plot) {
    std::ofstream fout("times.txt", std::ios_base::app);

    NeuralDensityOperators DensityOperators(N_v, N_h, N_a);

    double work_time = 0.0;
    TComplex* RoMatrix = DensityOperators.GetRoMatrix(&work_time, plot);

    std::cout << "N_v = " << N_v << ", N_h = " << N_h << ", N_a = " << N_a << "\n";
    std::cout << "Matrix size: " << N_v << "\n";
    std::cout << "Time: " << work_time << " s\n";
    std::cout << "Data tipe: " << TYPE_OUT << "\n\n";

    fout << "N_v = " << N_v << ", N_h = " << N_h << ", N_a = " << N_a << "\n";
    fout << "Matrix size: " << N_v << "\n";
    fout << "Time: " << work_time << " s\n";
    fout << "Data tipe: " << TYPE_OUT << "\n\n";

    delete[]RoMatrix;
}

