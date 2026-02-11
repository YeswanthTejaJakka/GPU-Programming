#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#define INF 1000000
__global__ void bfs_kernel(int n, int* row_ptr, int* col_idx, int* dist, int* pred, bool* changed) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p < n && dist[p] != INF) {
        for (int i = row_ptr[p]; i < row_ptr[p + 1]; i++) {
            int t = col_idx[i];
            if (dist[t] > (dist[p] + 1)) {
                dist[t] = dist[p] + 1;
                pred[t] = p;
                *changed = true;
            }
        }
    }
}

int main() {
    //Graph Setup (CSR Format)
    // Vertex 0 -> 1, 2
    // Vertex 1 -> 3
    // Vertex 2 -> 3, 4
    // Vertex 3 -> 4
    // Vertex 4 -> (none)
    int n = 5;
    int src = 0;

    int h_row_ptr[] = {0, 2, 3, 5, 6, 6};
    int h_col_idx[] = {1, 2, 3, 3, 4, 4};
    int num_edges = 6;

    // Initial distance/predecessor arrays
    std::vector<int> h_dist(n, INF);
    std::vector<int> h_pred(n, -1);
    h_dist[src] = 0; // Line 6: src.dist = 0

    // --- 2. GPU Memory Allocation ---
    int *d_row_ptr, *d_col_idx, *d_dist, *d_pred;
    bool *d_changed, h_changed;

    cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, num_edges * sizeof(int));
    cudaMalloc(&d_dist, n * sizeof(int));
    cudaMalloc(&d_pred, n * sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));

    cudaMemcpy(d_row_ptr, h_row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, h_dist.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pred, h_pred.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // --- 3. Execution (The While Loop) ---
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        bfs_kernel<<<blocks, threads>>>(n, d_row_ptr, d_col_idx, d_dist, d_pred, d_changed);

        cudaDeviceSynchronize();
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (h_changed);

    // --- 4. Results ---
    cudaMemcpy(h_dist.data(), d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pred.data(), d_pred, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Vertex\tDist\tPred\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t%d\t%d\n", i, h_dist[i], h_pred[i]);
    }

    // Cleanup
    cudaFree(d_row_ptr); cudaFree(d_col_idx);
    cudaFree(d_dist); cudaFree(d_pred); cudaFree(d_changed);

    return 0;
}
