#include "kernel.cuh"

namespace gf::simulator::single_dev::mix_block::kernel
{
    __global__ __launch_bounds__(1024) void 
    D3Q27Kernel(const D3Q27KernelParam __grid_constant__ param)
    {

    }
}