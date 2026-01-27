#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "ContactTable.hh"

// Simple micro-benchmark: compare host serial findOrInsert vs device kernel findOrInsert

__global__ void bench_findOrInsert_kernel(ContactHashTableView view, uint2* keys, uint* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    view.findOrInsert(keys[idx], out[idx]);
}

int main(int argc, char** argv)
{
    const int N = argc > 1 ? atoi(argv[1]) : 100000;
    const int capacity = N * 2;

    std::cout << "ContactHashTable benchmark N=" << N << " capacity=" << capacity << "\n";

    // Prepare keys
    std::vector<uint2> h_keys(N);
    for(int i = 0; i < N; ++i)
        h_keys[i] = make_uint2((unsigned)i, (unsigned)(i + 1));

    // ------------------ Host benchmark ------------------
    ContactHashTable<MemType::HOST> hostTable(capacity);

    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < N; ++i)
    {
        uint idx;
        hostTable.findOrInsert(h_keys[i], idx);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Host serial findOrInsert time (ms): " << host_ms << " ops/s: "
              << (N / (host_ms / 1000.0)) << "\n";

    // ------------------ Device benchmark (kernel only) ------------------
    ContactHashTable<MemType::DEVICE> devTable(capacity);
    ContactHashTableView view = devTable.getView();

    uint2* d_keys = nullptr;
    uint*  d_out  = nullptr;
    cudaMalloc(&d_keys, N * sizeof(uint2));
    cudaMalloc(&d_out, N * sizeof(uint));
    cudaMemcpy(d_keys, h_keys.data(), N * sizeof(uint2), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Warmup
    bench_findOrInsert_kernel<<<numBlocks, blockSize>>>(view, d_keys, d_out, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    bench_findOrInsert_kernel<<<numBlocks, blockSize>>>(view, d_keys, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    std::cout << "Device kernel findOrInsert time (ms): " << kernel_ms << " ops/s: "
              << (N / (kernel_ms / 1000.0)) << "\n";

    // Full pipeline timing (H2D + kernel + D2H)
    cudaDeviceSynchronize();
    auto p0 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_keys, h_keys.data(), N * sizeof(uint2), cudaMemcpyHostToDevice);
    bench_findOrInsert_kernel<<<numBlocks, blockSize>>>(view, d_keys, d_out, N);
    cudaMemcpy(d_out, d_out, N * sizeof(uint), cudaMemcpyDeviceToHost); // trivial copy to measure
    cudaDeviceSynchronize();
    auto p1 = std::chrono::high_resolution_clock::now();
    double pipeline_ms = std::chrono::duration<double, std::milli>(p1 - p0).count();
    std::cout << "Device pipeline time (H2D+kernel+D2H) ms: " << pipeline_ms << " ops/s: "
              << (N / (pipeline_ms / 1000.0)) << "\n";

    // Cleanup
    cudaFree(d_keys);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
