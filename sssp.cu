#include <stdio.h>
#include <cuda.h>
#include <limits.h>

__global__ void sssp_kernel(int n, int* w, int* row_ptr, int* col_idx, int* dist, bool* d_changed) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p >= n || dist[p] == INT_MAX) return;

    for (int i = row_ptr[p]; i < row_ptr[p+1]; i++) {
        int t = col_idx[i];
        int new_dist = dist[p] + w[i];

        // Use atomicMin to prevent race conditions during updates
        if (new_dist < atomicMin(&dist[t], new_dist)) {
            *d_changed = true;
        }
    }
}

int main() {
    int n = 5;
    int e = 5;
    int h_row_ptr[] = {0, 2, 3, 5, 6, 6};
    int h_col_idx[] = {1, 2, 3, 3, 4, 4};
    int h_weight[] = {1, 2, 3, 3, 4, 4};

    int *d_row_ptr, *d_col_idx, *d_weight, *d_dist;
    bool *d_changed, h_changed;

    cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, e * sizeof(int));
    cudaMalloc(&d_weight, e * sizeof(int));
    cudaMalloc(&d_dist, n * sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));

    // Initialize distances on host
    int h_dist[5];
    for(int i=0; i<n; i++) h_dist[i] = INT_MAX;
    h_dist[0] = 0; // Source is node 0

    cudaMemcpy(d_row_ptr, h_row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, e * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, e * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, h_dist, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int thread_blocks = (n + threads - 1) / threads;



    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
        sssp_kernel<<<thread_blocks, threads>>>(n, d_weight, d_row_ptr, d_col_idx, d_dist, d_changed);

        cudaDeviceSynchronize();
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (h_changed); // Keep going if any distance was updated

    cudaMemcpy(h_dist, d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Node:\t");
    for(int i=0; i<n; i++) printf("%d\t", i);
    printf("\nDist:\t");
    for(int i=0; i<n; i++) printf("%d\t", h_dist[i]);
    printf("\n");

    return 0;
}
