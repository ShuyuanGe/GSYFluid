#pragma once

#include <cstdint>
#include "vec3.hpp"
#include "config.hpp"

namespace gf::simulator::single_dev::mix_block::kernel
{
    struct D3Q27KernelParam
    {
        gf::basic::Vec3<std::int32_t> off;
        gf::basic::Vec3<std::int32_t> domDim;
        flag_t* blkFlagBuf = nullptr;
        real_t* glbRhoBuf  = nullptr;
        real_t* glbVxBuf   = nullptr;
        real_t* glbVyBuf   = nullptr;
        real_t* glbVzBuf   = nullptr;
        ddf_t*  srcDDFBuf  = nullptr;
        ddf_t*  dstDDFBuf  = nullptr;
        real_t* l2DDFSwapBuf = nullptr;
    };

    __global__ void D3Q27Kernel(const D3Q27KernelParam __grid_constant__ param);
}