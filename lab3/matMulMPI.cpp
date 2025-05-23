#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cassert>
#include <iomanip>
#include <string.h>
#include <sys/stat.h>

using namespace std;

bool file_exists(const string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

string make_filename(const char* prefix, int id, const char* suffix) {
    char buf[256];
    sprintf(buf, "%s%d%s", prefix, id, suffix);
    return string(buf);
}

vector< vector<double> > readMatrix(const char* filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    size_t rows, cols;
    file >> rows >> cols;

    vector< vector<double> > matrix(rows, vector<double>(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            file >> matrix[i][j];

    file.close();
    return matrix;
}

void writeMatrix(const char* filename, const vector< vector<double> >& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    size_t rows = matrix.size(), cols = matrix[0].size();
    file << rows << " " << cols << endl;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j)
            file << setprecision(6) << matrix[i][j] << " ";
        file << endl;
    }

    file.close();
}

void writeTimeAndSize(const char* filename, double time_sec, size_t rowsA, size_t colsA, size_t rowsB, size_t colsB) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file " << filename << endl;
        exit(EXIT_FAILURE);
    }
    file << "Matrix A: " << rowsA << "x" << colsA << endl;
    file << "Matrix B: " << rowsB << "x" << colsB << endl;
    file << "Execution Time (s): " << time_sec << endl;
    file.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    mkdir("results", 0777);

    int test_id = 1;
    while (true) {
        string fileA = make_filename("matrices/matrixA_", test_id, ".txt");
        string fileB = make_filename("matrices/matrixB_", test_id, ".txt");
        string fileC = make_filename("results/matrixC_", test_id, ".txt");
        string fileTime = make_filename("results/time_", test_id, ".txt");

        if (rank == 0 && (!file_exists(fileA) || !file_exists(fileB))) {
            cout << "Finished processing all available matrix files." << endl;
            break;
        }

        size_t rowsA, colsA, rowsB, colsB;
        vector< vector<double> > A, B;

        if (rank == 0) {
            cout << "Processing Test " << test_id << ": " << fileA << " x " << fileB << endl;
            A = readMatrix(fileA.c_str());
            B = readMatrix(fileB.c_str());

            rowsA = A.size();
            colsA = A[0].size();
            rowsB = B.size();
            colsB = B[0].size();
        }

        MPI_Bcast(&rowsA, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&colsA, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&rowsB, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&colsB, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            B.resize(rowsB, vector<double>(colsB));
        }

        for (size_t i = 0; i < rowsB; ++i)
            MPI_Bcast(&B[i][0], colsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        size_t rows_per_proc = rowsA / size;
        size_t remainder = rowsA % size;
        size_t local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
        size_t start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);

        vector< vector<double> > localA(local_rows, vector<double>(colsA));

        if (rank == 0) {
            for (int p = 0; p < size; ++p) {
                size_t p_rows = rows_per_proc + (p < remainder ? 1 : 0);
                size_t p_start = p * rows_per_proc + (p < remainder ? p : remainder);

                if (p == 0) {
                    for (size_t i = 0; i < p_rows; ++i)
                        localA[i] = A[p_start + i];
                } else {
                    for (size_t i = 0; i < p_rows; ++i)
                        MPI_Send(&A[p_start + i][0], colsA, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            for (size_t i = 0; i < local_rows; ++i)
                MPI_Recv(&localA[i][0], colsA, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double start_time = MPI_Wtime();

        vector< vector<double> > localC(local_rows, vector<double>(colsB, 0.0));
        for (size_t i = 0; i < local_rows; ++i)
            for (size_t j = 0; j < colsB; ++j)
                for (size_t k = 0; k < colsA; ++k)
                    localC[i][j] += localA[i][k] * B[k][j];

        double end_time = MPI_Wtime();

        if (rank == 0) {
            vector< vector<double> > C(rowsA, vector<double>(colsB));
            for (size_t i = 0; i < local_rows; ++i)
                C[i] = localC[i];

            for (int p = 1; p < size; ++p) {
                size_t p_rows = rows_per_proc + (p < remainder ? 1 : 0);
                size_t p_start = p * rows_per_proc + (p < remainder ? p : remainder);

                for (size_t i = 0; i < p_rows; ++i)
                    MPI_Recv(&C[p_start + i][0], colsB, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            writeMatrix(fileC.c_str(), C);
            writeTimeAndSize(fileTime.c_str(), end_time - start_time, rowsA, colsA, rowsB, colsB);

            cout << "Execution time: " << end_time - start_time << " seconds." << endl;
        } else {
            for (size_t i = 0; i < local_rows; ++i)
                MPI_Send(&localC[i][0], colsB, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }

        test_id++;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
