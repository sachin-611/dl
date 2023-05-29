#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <thread>

#define TILE_WIDTH 32

// Kernel function for matrix multiplication on the GPU
__global__ void matrixMult(int *a, int *b, int *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main()
{
    int n = 1600;

    // Allocate memory for matrices A, B, and C
    int *a = new int[n * n];
    int *b = new int[n * n];
    int *c = new int[n * n];
    
    // Generate random values for matrices A and B
    std::srand(std::time(0));
    for (int i = 0; i < n * n; ++i) {
        a[i] = std::rand() % 10;
        b[i] = std::rand() % 10;
    }

    // Variables for GPU memory
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, n * n * sizeof(int));
    cudaMalloc(&dev_b, n * n * sizeof(int));
    cudaMalloc(&dev_c, n * n * sizeof(int));

    // Copy matrices A and B from host to device
    cudaMemcpy(dev_a, a, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for kernel launch
    dim3 dimGrid((n - 1) / TILE_WIDTH + 1, (n - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Start timer for parallel execution
    clock_t parallel_start = clock();

    // Launch the matrix multiplication kernel on the GPU
    matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, n);

    // Wait for GPU to finish execution
    cudaDeviceSynchronize();

    // End timer for parallel execution
    clock_t parallel_end = clock();
    double parallel_time = double(parallel_end - parallel_start) / CLOCKS_PER_SEC;

    // Copy the result matrix C from device to host
    cudaMemcpy(c, dev_c, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print matrices A, B, and the result matrix C
    std::cout << "A matrix:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << a[i * n + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "B matrix:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << b[i * n + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout <<"Result matrix:\n";
    for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
    std::cout << c[i * n + j] << " ";
    }
    std::cout << "\n";
    }

    // Start timer for serial execution
    clock_t serial_start = clock();
    // Perform matrix multiplication on the CPU (serial execution)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    // End timer for serial execution
    clock_t serial_end = clock();
    double serial_time = double(serial_end - serial_start) / CLOCKS_PER_SEC;

    // Print execution times
    printf("Parallel execution time: : %3.7f ms\n", parallel_time);
    printf("Serial execution time: : %3.7f ms\n", serial_time);

    // Free GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Free CPU memory
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;

}
