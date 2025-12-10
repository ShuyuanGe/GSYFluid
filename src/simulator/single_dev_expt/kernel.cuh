#pragma once

#include <cuda_runtime.h>
#include "local_config.hpp"

namespace gf::simulator::single_dev_expt
{
    struct HaloBlockingL1L2Param
    {
        real_t invTau = 0;
        idx_t nloop = 0;
        idx_t offx = 0, offy = 0, offz = 0;
        idx_t glbnx = 0, glbny = 0, glbnz = 0;
        const flag_t* blkFlagBuf = nullptr;
        real_t* glbRhoBuf = nullptr;
        real_t* glbVxBuf = nullptr;
        real_t* glbVyBuf = nullptr;
        real_t* glbVzBuf = nullptr;
        ddf_t* glbSrcDDFBuf = nullptr;
        ddf_t* glbDstDDFBuf = nullptr;
        real_t* glbSwapDDFBuf = nullptr;
    };

    __global__ void HaloBlockingL1L2D3Q27PullKernel(const HaloBlockingL1L2Param __grid_constant__ param);
    __global__ void HaloBlockingL1L2D3Q27DumpKernel(const HaloBlockingL1L2Param __grid_constant__ param);
}