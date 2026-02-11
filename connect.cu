#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define NUM_VERTICES 8
#define NUM_EDGES 10

// CUDA kernel for label propagation
// Each vertex looks at its neighbors; if a neighbor has a smaller component ID,
// the vertex adopts it. This continues until no more changes occur.
__global__ void find_components_kernel(int* d_src, int* d_dst, int* d_comp, int num_edges, bool* d_changed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        int u = d_src[tid];
        int v = d_dst[tid];

        // If neighbor has a smaller component ID, take it
        if (d_comp[u] < d_comp[v]) {
            atomicMin(&d_comp[v], d_comp[u]);
            *d_changed = true;
        } else if (d_comp[v] < d_comp[u]) {
            atomicMin(&d_comp[u], d_comp[v]);
            *d_changed = true;
        }
    }
}

int main() {
    // Example Graph (Undirected)
    int h_src[NUM_EDGES] = {0, 1, 2, 4, 5, 0, 2, 3, 4, 6};
    int h_dst[NUM_EDGES] = {1, 2, 0, 5, 6, 3, 3, 0, 6, 4};

    // Component IDs: Initially, every vertex is its own component
    int h_comp[NUM_VERTICES];
    for (int i = 0; i < NUM_VERTICES; i++) h_comp[i] = i;

    int *d_src, *d_dst, *d_comp;
    bool *d_changed, h_changed;

    // Allocate Memory
    cudaMalloc(&d_src, NUM_EDGES * sizeof(int));
    cudaMalloc(&d_dst, NUM_EDGES * sizeof(int));
    cudaMalloc(&d_comp, NUM_VERTICES * sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));

    // Copy data to device
    cudaMemcpy(d_src, h_src, NUM_EDGES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, NUM_EDGES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_comp, h_comp, NUM_VERTICES * sizeof(int), cudaMemcpyHostToDevice);

    // Iterate until convergence (no more component ID updates)
    int iterations = 0;
    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (NUM_EDGES + threadsPerBlock - 1) / threadsPerBlock;

        find_components_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, d_comp, NUM_EDGES, d_changed);

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        iterations++;
    } while (h_changed);

    // Copy result back
    cudaMemcpy(h_comp, d_comp, NUM_VERTICES * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Algorithm converged in " << iterations << " iterations.\n";
    for (int i = 0; i < NUM_VERTICES; i++) {
        std::cout << "Vertex " << i << " is in Component " << h_comp[i] << std::endl;
    }

    // Cleanup
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_comp); cudaFree(d_changed);
    return 0;
}
