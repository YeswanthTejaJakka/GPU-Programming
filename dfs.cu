#include <cuda_runtime.h>
#include <stdio.h>

#define WHITE 0
#define GRAY  1
#define BLACK 2

// Structure to mimic the Vertex properties in your pseudocode
struct Vertex {
    int color;
    int start;
    int etime;
    int pred;
};

// Global clock in device memory
__device__ int g_clock = 0;

__global__ void dfs_kernel(int n, int* row_ptr, int* col_idx, Vertex* vertices, int start_node) {
    // Note: This implementation uses a single thread to manage the "Traverse" logic
    // for one component. DFS is fundamentally difficult to parallelize.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) { // Only one thread starts the traversal to maintain DFS order
        int stack[1024];
        int top = -1;

        // Push initial node
        stack[++top] = start_node;

        while (top >= 0) {
            int p = stack[top];

            if (vertices[p].color == WHITE) {
                vertices[p].start = atomicAdd(&g_clock, 1) + 1; // p.start = ++clock
                vertices[p].color = GRAY;

                // Push neighbors in reverse to maintain correct DFS order
                for (int i = row_ptr[p + 1] - 1; i >= row_ptr[p]; i--) {
                    int t = col_idx[i];
                    if (vertices[t].color == WHITE) {
                        vertices[t].pred = p;
                        stack[++top] = t;
                    }
                }
            } else if (vertices[p].color == GRAY) {
                // All neighbors visited, finish node
                vertices[p].color = BLACK;
                vertices[p].etime = atomicAdd(&g_clock, 1) + 1; // p.etime = ++clock
                top--;
            } else {
                top--; // Already BLACK
            }
        }
    }
}

int main() {
    int n = 5;
    // Example graph in CSR format
    int h_row_ptr[] = {0, 2, 3, 5, 6, 6};
    int h_col_idx[] = {1, 2, 3, 3, 4, 4};

    Vertex* d_vertices;
    int *d_row_ptr, *d_col_idx;

    cudaMalloc(&d_vertices, n * sizeof(Vertex));
    cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, 6 * sizeof(int));

    // Initialize vertices
    Vertex h_vertices[5];
    for(int i=0; i<5; i++) {
        h_vertices[i] = {WHITE, -1, -1, -1};
    }

    cudaMemcpy(d_vertices, h_vertices, n * sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, 6 * sizeof(int), cudaMemcpyHostToDevice);

    // Call kernel for each component
    for (int i = 0; i < n; i++) {
        // In a real GPU implementation, you'd check color on host then launch
        dfs_kernel<<<1, 1>>>(n, d_row_ptr, d_col_idx, d_vertices, i);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_vertices, d_vertices, n * sizeof(Vertex), cudaMemcpyDeviceToHost);

    printf("Node\tStart\tEnd\tPred\n");
    for(int i=0; i<n; i++) {
        printf("%d\t%d\t%d\t%d\n", i, h_vertices[i].start, h_vertices[i].etime, h_vertices[i].pred);
    }

    return 0;
}
