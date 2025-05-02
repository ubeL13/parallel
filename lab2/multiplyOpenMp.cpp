#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <omp.h>

using namespace std;

bool file_exists(const string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

vector<vector<double>> readMatrix(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    size_t rows, cols;
    file >> rows >> cols;
    vector<vector<double>> matrix(rows, vector<double>(cols));

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            file >> matrix[i][j];

    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    size_t rows = matrix.size(), cols = matrix[0].size();
    file << rows << " " << cols << endl;
    for (const auto& row : matrix) {
        for (const auto& val : row)
            file << setprecision(6) << val << " ";
        file << endl;
    }
}

void writeTimeAndSize(const string& filename, double time_sec, size_t rowsA, size_t colsA, size_t rowsB, size_t colsB) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file " << filename << endl;
        exit(EXIT_FAILURE);
    }
    file << "Matrix A: " << rowsA << "x" << colsA << endl;
    file << "Matrix B: " << rowsB << "x" << colsB << endl;
    file << "Execution Time (s): " << time_sec << endl;
}

vector<vector<double>> multiplyMatrices(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    size_t n = A.size(), m = A[0].size(), p = B[0].size();
    assert(m == B.size());

    vector<vector<double>> C(n, vector<double>(p, 0.0));

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < m; ++k)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }

    return C;
}

int main() {
    system("mkdir -p results");

    // Установка числа потоков под твой AMD Ryzen 5 5500U
    omp_set_num_threads(12);
    cout << "OpenMP: using " << omp_get_max_threads() << " threads" << endl;

    int test_id = 1;
    while (true) {
        string fileA = "../matrices/matrixA_" + to_string(test_id) + ".txt";
        string fileB = "../matrices/matrixB_" + to_string(test_id) + ".txt";
        string fileC = "results/matrixC_" + to_string(test_id) + ".txt";
        string fileTime = "results/time_" + to_string(test_id) + ".txt";

        if (!file_exists(fileA) || !file_exists(fileB)) {
            cout << "Finished processing all available matrix files." << endl;
            break;
        }

        auto A = readMatrix(fileA);
        auto B = readMatrix(fileB);

        cout << "Processing Test " << test_id << ": " << fileA << " x " << fileB << endl;

        auto start = chrono::high_resolution_clock::now();
        auto C = multiplyMatrices(A, B);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> elapsed = end - start;
        cout << "Execution time: " << elapsed.count() << " seconds." << endl;

        writeMatrix(fileC, C);
        writeTimeAndSize(fileTime, elapsed.count(), A.size(), A[0].size(), B.size(), B[0].size());

        test_id++;
    }

    return 0;
}
