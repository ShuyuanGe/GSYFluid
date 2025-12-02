#include <format>
#include <iostream>
#include <cuda_runtime.h>
#include "cu_exception.cuh"

int main()
{
    try
    {
        cudaDeviceProp prop;
        CU_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << std::format("# SM = {}", prop.multiProcessorCount) << std::endl;
        std::cout << std::format("max thread per block = {}", prop.maxThreadsPerBlock) << std::endl;
        std::cout << std::format("shared memory = {} KB", prop.sharedMemPerBlockOptin / 1024) << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        exit(1);
    }
    
    return 0;
}