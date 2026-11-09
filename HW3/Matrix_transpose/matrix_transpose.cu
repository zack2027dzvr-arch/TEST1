#define PPROG_IMPLEMENTATION

#include <random.cuh>
#include <timer.cuh>
#include <utils.cuh>

#include <chrono>
#include <thread>

// ===================================================================================
// CONFIGURATION

#define N 5000 /*1 << 8*/
#define TILE_SIZE 32

// ===================================================================================
// UTILITIES

#define cuda_check_error()                                                     \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

// NOTE: use this function to visualize the matrices
void print_matrix(const std::vector<int>& matrix) {
    printf("\n");
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            printf("%3d ", matrix[y * N + x]);
        }
        printf("\n");
    }
}

// ===================================================================================
// @@ IMPLEMENTATION

//@@ Write sequential matrix transpose
std::vector<int> transpose_cpu(const std::vector<int>& input, int n) {
    std::vector<int> output{};
    output.resize(input.size());

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            output[col * n + row] = input[row * n + col];
        }
    }

    return output;
}

//@@ Write the transpose kernel
__global__ void transpose_gpu(int* input, int* output, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

// ===================================================================================

int main(int argc, char** argv) {

    // Generates random matrix data
    printf("generating random data...\n");
    auto matrix = random_matrix<int>(N, N, 2000 /* SEED */);
    printf("done\n");

    std::vector<int> result_cpu{};
    result_cpu.resize(matrix.size());

    std::vector<int> result_gpu{};
    result_gpu.resize(matrix.size());

    //print_matrix(matrix);

    // Allocate matrix in GPU memory
    int *d_matrix;
    cudaMalloc((void**)&d_matrix, matrix.size() * sizeof(int));
    cudaMemcpy(d_matrix, matrix.data(), matrix.size() * sizeof(int), cudaMemcpyHostToDevice);
    cuda_check_error();

    // Create result matrix
    int *d_result;
    cudaMalloc(&d_result, matrix.size() * sizeof(int));
    cuda_check_error();

    // Setup blocks and threads count
    dim3 threads = dim3(TILE_SIZE, TILE_SIZE);
    dim3 blocks = dim3(
        ((N) + TILE_SIZE - 1) / TILE_SIZE,
        ((N) + TILE_SIZE - 1) / TILE_SIZE
    );

    // ===================================================================================
    // SEQUENTIAL

    auto timer_cpu = timerCPU{};
    timer_cpu.start();

    // Sequential
    result_cpu = transpose_cpu(matrix, N);

    timer_cpu.stop();

    //print_matrix(result_cpu);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ===================================================================================
    // GPU

    auto timer_gpu = timerGPU{};
    timer_gpu.start();

    // Kernel launch
    transpose_gpu<<<blocks, threads>>>(d_matrix, d_result, N);

    timer_gpu.stop();

    // Move result matrix to CPU memory
    cudaMemcpy(result_gpu.data(), d_result, result_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check_error();

    //print_matrix(result_gpu);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ===================================================================================
    // TIMERS

    auto cpu_ms = timer_cpu.elapsed_ms();
    auto gpu_ms = timer_gpu.elapsed_ms();

    printf("transpose CPU\n\t%f ms\n", cpu_ms);

    printf("transpose GPU\n\t%f ms (speedup: cpu %.2fx)\n",
        gpu_ms, cpu_ms / gpu_ms);

    // ===================================================================================
    // CHECK

    bool ok_result = mat_check_result(result_cpu, result_gpu, N);
    printf("Solution CPU vs GPU: %s\n", ok_result ? "CORRECT" : "INCORRECT");

    // ===================================================================================

    // Free cuda memory
    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}
