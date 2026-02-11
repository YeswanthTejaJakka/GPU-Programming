#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define IN 1e9
#define INF (int)IN

__global__ void floyd_warshall_kernel(float* dist, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        float ik = dist[i * n + k];
        float kj = dist[k * n + j];
        float ij = dist[i * n + j];

        if (ik != INF && kj != INF && ik + kj < ij) {
            dist[i * n + j] = ik + kj;
        }
    }
}

int main() {
    int n = 4;        // Example: 4 vertices
    size_t size = n * n * sizeof(float);


    int  h_dist[] = {
        0,   3,   INF, 7,
        8,   0,   2,   INF,
        5,   INF, 0,   1,
        2,   INF, INF, 0
    };

    float *d_dist;
    cudaMalloc(&d_dist, size);
    cudaMemcpy(d_dist, h_dist ,size, cudaMemcpyHostToDevice);

    // Execution configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + 15) / 16, (n + 15) / 16);

    // Outer loop runs on CPU
    for (int k = 0; k < n; k++) {
        floyd_warshall_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_dist, k, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_dist, d_dist, size, cudaMemcpyDeviceToHost);

    // Printing  result
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (h_dist[i * n + j] == INF) printf("INF ");
            else printf("%d ", h_dist[i * n + j]);
        }
        printf("\n");
    }

    cudaFree(d_dist);
    return 0;
}
