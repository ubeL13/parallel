#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <cuda_runtime.h>

using namespace std;

// Проверка существования файла
bool file_exists(const string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

// Чтение матрицы из файла
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

// Запись матрицы в файл
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

// Запись времени выполнения и размеров матриц
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

// Ядро CUDA для умножения матриц
__global__ void matrixMultiplyKernel(double* A, double* B, double* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        double sum = 0.0;
        for (int k = 0; k < m; k++) {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}   

// Умножение матриц с использованием CUDA
vector<vector<double>> multiplyMatricesCUDA(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    size_t n = A.size(), m = A[0].size(), p = B[0].size();
    assert(m == B.size());

    // Преобразование матриц в плоские массивы
    double* flatA = new double[n * m];
    double* flatB = new double[m * p];
    double* flatC = new double[n * p];

    // Заполнение flatA и flatB
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            flatA[i * m + j] = A[i][j];

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < p; ++j)
            flatB[i * p + j] = B[i][j];

    // Выделение памяти на GPU
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * m * sizeof(double));
    cudaMalloc(&d_B, m * p * sizeof(double));
    cudaMalloc(&d_C, n * p * sizeof(double));

    // Копирование данных на GPU
    cudaMemcpy(d_A, flatA, n * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB, m * p * sizeof(double), cudaMemcpyHostToDevice);

    // Настройка параметров запуска ядра
    dim3 blockSize(16, 16);
    dim3 gridSize((p + blockSize.x - 1) / blockSize.x, 
                 (n + blockSize.y - 1) / blockSize.y);

    // Запуск ядра
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n, m, p);

    // Проверка ошибок CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel error: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }

    // Копирование результата обратно на CPU
    cudaMemcpy(flatC, d_C, n * p * sizeof(double), cudaMemcpyDeviceToHost);

    // Преобразование результата в vector<vector<double>>
    vector<vector<double>> C(n, vector<double>(p));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < p; ++j)
            C[i][j] = flatC[i * p + j];

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] flatA;
    delete[] flatB;
    delete[] flatC;

    return C;
}

int main() {
    system("mkdir -p results");
    
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
        auto C = multiplyMatricesCUDA(A, B);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> elapsed = end - start;
        cout << "Execution time: " << elapsed.count() << " seconds." << endl;

        writeMatrix(fileC, C);
        writeTimeAndSize(fileTime, elapsed.count(), A.size(), A[0].size(), B.size(), B[0].size());

        test_id++;
    }

    return 0;
}