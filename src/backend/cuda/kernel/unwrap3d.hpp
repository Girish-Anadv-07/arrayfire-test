/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <kernel/config.hpp>
#include <nvrtc_kernel_headers/unwrap_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T>
void unwrap3d(Param<T> out, CParam<T> in, const int wx, const int wy, const int wz,
            const int sx, const int sy, const int sz, const int px, const int py,
            const int pz, const int dx, const int dy, const int dz, const int nx, 
            const bool is_column) {
    auto unwrap = common::getKernel(
        "arrayfire::cuda::unwrap", {{unwrap_cuh_src}},
        TemplateArgs(TemplateTypename<T>t(), TemplateArg(is_column)));

    dim3 threads, blocks;
    int reps;

    if (is_column) {
        int TX = std::min(THREADS_PER_BLOCK, nextpow2(out.dims[0]));
        int TY = std::min(THREADS_PER_BLOCK / TX, nextpow2(out.dims[1]));

        threads = dim3(TX, TY, THREADS_PER_BLOCK / (TX * TY));
        blocks = dim3(divup(out.dims[1], threads.y), divup(out.dims[2], threads.z), 1);
        reps   = divup((wx * wy * wz),
                       threads.x);  // is > 1 only when TX == 256 && wx * wy > 256
    } else {
        threads = dim3(THREADS_X, THREADS_Y);
        blocks = dim3(divup(out.dims[0], threads.x), divup(out.dims[2], threads.z), 1);

        reps = divup((wx * wy * wz), threads.y);
    }

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    unwrap3d(qArgs, out, in, wx, wy, wz, sx, sy, sz, px, py, pz, dx, dy, dz, nx, reps);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
