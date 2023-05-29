#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
using namespace std;
void serial_add(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void cuda_add(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 500000;
    int a[n], b[n], c[n], d[n];
    int *dev_a, *dev_b, *dev_c;

    // Allocate memory on the device
    cudaMalloc(&dev_a, n * sizeof(int));
    cudaMalloc(&dev_b, n * sizeof(int));
    cudaMalloc(&dev_c, n * sizeof(int));

    // Initialize arrays on the host
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Copy arrays from host to device
    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    // Perform CUDA addition and measure time
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize; // round up to nearest integer
    auto start_cuda = chrono::high_resolution_clock::now();
    cuda_add<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c, n);
    cudaMemcpy(c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    auto end_cuda = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_cuda = end_cuda - start_cuda;

    // Perform serial addition and measure time
    auto start_serial = chrono::high_resolution_clock::now();
    serial_add(a, b, d, n);
    auto end_serial = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_serial = end_serial - start_serial;

    // Verify results and print execution time
	//for(int i=0;i<n;i++)
	//	cout<<c[i]<<","<<d[i]<<" ";
	
    cout << "\nSerial execution time: " << elapsed_serial.count()*1000 << " milliseconds" << endl;
    cout << "CUDA execution time: " << elapsed_cuda.count()*1000 << " milliseconds" << endl;

    // Free memory on the device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
