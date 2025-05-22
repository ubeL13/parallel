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
#include <cstdio>

using namespace std;

int main() {
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    cout << "Max threads per block: " << maxThreadsPerBlock << endl;
}