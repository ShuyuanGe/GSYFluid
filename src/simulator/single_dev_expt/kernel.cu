#include "kernel.cuh"
#include "L1L2StreamCore.cuh"
#include "device_function.hpp"



namespace gf::simulator::single_dev_expt
{
    __global__ __launch_bounds__(1024) void
    HaloBlockingL1L2D3Q27PullKernel(const HaloBlockingL1L2Param __grid_constant__ param)
    {
        extern __shared__ real_t blkDDFBuf [];

        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blknz = gridDim.z * blockDim.z;
        const idx_t blki = (blkny * blkz + blky) * blknx + blkx;

        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;

        const flag_t flagi = param.blkFlagBuf[blki];
        real_t rhoi = 1, vxi = 0, vyi = 0, vzi = 0;

        real_t fni[27];

        //first global load
        if((flagi & LOAD_DDF_BIT)!=0)
        {
            const idx_t glbndx = (glbx==0) ? 0 : -1;
            const idx_t glbpdx = (glbx==param.glbnx-1) ? 0 : 1;
            const idx_t glbndy = (glby==0) ? 0 : -param.glbnx;
            const idx_t glbpdy = (glby==param.glbny-1) ? 0 : param.glbnx;
            const idx_t glbndz = (glbz==0) ? 0 : -param.glbnx * param.glbny;
            const idx_t glbpdz = (glbz==param.glbnz-1) ? 0 : param.glbnx * param.glbny;

            //f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.glbSrcDDFBuf[ 0*glbn+glbi+glbpdx+glbpdy+glbpdz];
            //f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.glbSrcDDFBuf[ 1*glbn+glbi       +glbpdy+glbpdz];
            //f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
            fni[ 2] = param.glbSrcDDFBuf[ 2*glbn+glbi+glbndx+glbpdy+glbpdz];
            //f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.glbSrcDDFBuf[ 3*glbn+glbi+glbpdx       +glbpdz];
            //f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.glbSrcDDFBuf[ 4*glbn+glbi              +glbpdz];
            //f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
            fni[ 5] = param.glbSrcDDFBuf[ 5*glbn+glbi+glbndx       +glbpdz];
            //f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
            fni[ 6] = param.glbSrcDDFBuf[ 6*glbn+glbi+glbpdx+glbndy+glbpdz];
            //f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
            fni[ 7] = param.glbSrcDDFBuf[ 7*glbn+glbi       +glbndy+glbpdz];
            //f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
            fni[ 8] = param.glbSrcDDFBuf[ 8*glbn+glbi+glbndx+glbndy+glbpdz];

            //f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.glbSrcDDFBuf[ 9*glbn+glbi+glbpdx+glbpdy       ];
            //f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.glbSrcDDFBuf[10*glbn+glbi       +glbpdy       ];
            //f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
            fni[11] = param.glbSrcDDFBuf[11*glbn+glbi+glbndx+glbpdy       ];
            //f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.glbSrcDDFBuf[12*glbn+glbi+glbpdx              ];
            //f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
            fni[13] = param.glbSrcDDFBuf[13*glbn+glbi                     ];
            //f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
            fni[14] = param.glbSrcDDFBuf[14*glbn+glbi+glbndx              ];
            //f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
            fni[15] = param.glbSrcDDFBuf[15*glbn+glbi+glbpdx+glbndy       ];
            //f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
            fni[16] = param.glbSrcDDFBuf[16*glbn+glbi       +glbndy       ];
            //f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
            fni[17] = param.glbSrcDDFBuf[17*glbn+glbi+glbndx+glbndy       ];

            //f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
            fni[18] = param.glbSrcDDFBuf[18*glbn+glbi+glbpdx+glbpdy+glbndz];
            //f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
            fni[19] = param.glbSrcDDFBuf[19*glbn+glbi       +glbpdy+glbndz];
            //f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
            fni[20] = param.glbSrcDDFBuf[20*glbn+glbi+glbndx+glbpdy+glbndz];
            //f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
            fni[21] = param.glbSrcDDFBuf[21*glbn+glbi+glbpdx       +glbndz];
            //f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
            fni[22] = param.glbSrcDDFBuf[22*glbn+glbi              +glbndz];
            //f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
            fni[23] = param.glbSrcDDFBuf[23*glbn+glbi+glbndx       +glbndz];
            //f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
            fni[24] = param.glbSrcDDFBuf[24*glbn+glbi+glbpdx+glbndy+glbndz];
            //f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
            fni[25] = param.glbSrcDDFBuf[25*glbn+glbi       +glbndy+glbndz];
            //f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
            fni[26] = param.glbSrcDDFBuf[26*glbn+glbi+glbndx+glbndy+glbndz];
        }

        // if((flagi & EQU_DDF_BIT)!=0)
        // {
        //     rhoi    = param.glbRhoBuf[glbi];
        //     vxi     = param.glbVxBuf[glbi];
        //     vyi     = param.glbVyBuf[glbi];
        //     vzi     = param.glbVzBuf[glbi];
        //     gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
        // }

        // if((flagi & BOUNCE_BACK_BIT)!=0)
        // {
        //     std::swap(fni[ 0], fni[26]);
        //     std::swap(fni[ 1], fni[25]);
        //     std::swap(fni[ 2], fni[24]);
        //     std::swap(fni[ 3], fni[23]);
        //     std::swap(fni[ 4], fni[22]);
        //     std::swap(fni[ 5], fni[21]);
        //     std::swap(fni[ 6], fni[20]);
        //     std::swap(fni[ 7], fni[19]);
        //     std::swap(fni[ 8], fni[18]);
        //     std::swap(fni[ 9], fni[17]);
        //     std::swap(fni[10], fni[16]);
        //     std::swap(fni[11], fni[15]);
        //     std::swap(fni[12], fni[14]);
        // }

        // if((flagi & COLLIDE_BIT)!=0)
        // {
        //     gf::lbm_core::bgk::collision<27>(param.invTau, rhoi, vxi, vyi, vzi, std::begin(fni));
        // }

        #pragma unroll
        for(idx_t iter=1 ; iter<param.nloop ; ++iter)
        {
            gf::core::L1L2StreamCore::StreamCore3D<27>::stream(std::begin(fni), blkDDFBuf, param.glbSwapDDFBuf);

            // if((flagi & EQU_DDF_BIT)!=0)
            // {
            //     gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
            // }

            // if((flagi & BOUNCE_BACK_BIT)!=0)
            // {
            //     std::swap(fni[ 0], fni[26]);
            //     std::swap(fni[ 1], fni[25]);
            //     std::swap(fni[ 2], fni[24]);
            //     std::swap(fni[ 3], fni[23]);
            //     std::swap(fni[ 4], fni[22]);
            //     std::swap(fni[ 5], fni[21]);
            //     std::swap(fni[ 6], fni[20]);
            //     std::swap(fni[ 7], fni[19]);
            //     std::swap(fni[ 8], fni[18]);
            //     std::swap(fni[ 9], fni[17]);
            //     std::swap(fni[10], fni[16]);
            //     std::swap(fni[11], fni[15]);
            //     std::swap(fni[12], fni[14]);
            // }

            // if((flagi & COLLIDE_BIT)!=0)
            // {
            //     gf::lbm_core::bgk::collision<27>(param.invTau, rhoi, vxi, vyi, vzi, std::begin(fni));
            // }
        }

        //last global store
        if((flagi & (STORE_DDF_BIT | CORRECT_BIT))==(STORE_DDF_BIT | CORRECT_BIT))
        {
            param.glbDstDDFBuf[ 0*glbn+glbi] = fni[ 0];
            param.glbDstDDFBuf[ 1*glbn+glbi] = fni[ 1];
            param.glbDstDDFBuf[ 2*glbn+glbi] = fni[ 2];
            param.glbDstDDFBuf[ 3*glbn+glbi] = fni[ 3];
            param.glbDstDDFBuf[ 4*glbn+glbi] = fni[ 4];
            param.glbDstDDFBuf[ 5*glbn+glbi] = fni[ 5];
            param.glbDstDDFBuf[ 6*glbn+glbi] = fni[ 6];
            param.glbDstDDFBuf[ 7*glbn+glbi] = fni[ 7];
            param.glbDstDDFBuf[ 8*glbn+glbi] = fni[ 8];

            param.glbDstDDFBuf[ 9*glbn+glbi] = fni[ 9];
            param.glbDstDDFBuf[10*glbn+glbi] = fni[10];
            param.glbDstDDFBuf[11*glbn+glbi] = fni[11];
            param.glbDstDDFBuf[12*glbn+glbi] = fni[12];
            param.glbDstDDFBuf[13*glbn+glbi] = fni[13];
            param.glbDstDDFBuf[14*glbn+glbi] = fni[14];
            param.glbDstDDFBuf[15*glbn+glbi] = fni[15];
            param.glbDstDDFBuf[16*glbn+glbi] = fni[16];
            param.glbDstDDFBuf[17*glbn+glbi] = fni[17];

            param.glbDstDDFBuf[18*glbn+glbi] = fni[18];
            param.glbDstDDFBuf[19*glbn+glbi] = fni[19];
            param.glbDstDDFBuf[20*glbn+glbi] = fni[20];
            param.glbDstDDFBuf[21*glbn+glbi] = fni[21];
            param.glbDstDDFBuf[22*glbn+glbi] = fni[22];
            param.glbDstDDFBuf[23*glbn+glbi] = fni[23];
            param.glbDstDDFBuf[24*glbn+glbi] = fni[24];
            param.glbDstDDFBuf[25*glbn+glbi] = fni[25];
            param.glbDstDDFBuf[26*glbn+glbi] = fni[26];

            param.glbRhoBuf[glbi] = rhoi;
            param.glbVxBuf[glbi] = vxi;
            param.glbVyBuf[glbi] = vyi;
            param.glbVzBuf[glbi] = vzi;
        }
    }
}