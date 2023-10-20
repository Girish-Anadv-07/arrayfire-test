/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>

#include <common/dispatch.hpp>  // common utility header for CUDA & OpenCL backends
                                // has the divup macro

#include <debug_cuda.hpp>  // For Debug only related CUDA validations

#include <common/kernel_cache.hpp>  // nvrtc cache mechanims API

#include <nvrtc_kernel_headers/exampleFunction_cuh.hpp>  //kernel generated by nvrtc

namespace arrayfire {
namespace cuda {

namespace kernel {

static const unsigned TX = 16;  // Kernel Launch Config Values
static const unsigned TY = 16;  // Kernel Launch Config Values

template<typename T>  // CUDA kernel wrapper function
void exampleFunc(Param<T> c, CParam<T> a, CParam<T> b, const af_someenum_t p) {
    auto exampleFunc = common::getKernel("arrayfire::cuda::exampleFunc",
                                         {{exampleFunction_cuh_src}},
                                         TemplateArgs(TemplateTypename<T>()));

    dim3 threads(TX, TY, 1);  // set your cuda launch config for blocks

    int blk_x = divup(c.dims[0], threads.x);
    int blk_y = divup(c.dims[1], threads.y);
    dim3 blocks(blk_x, blk_y);  // set your cuda launch config for grid

    // EnqueueArgs encapsulates CUDA kernel launch
    // configuration paramters. There are various versions
    // of EnqueueArgs constructors that you can use depending
    // on your CUDA kernels needs such as shared memory etc.
    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    // Call the kernel functor retrieved using arrayfire::common::getKernel
    exampleFunc(qArgs, c, a, b, p);

    POST_LAUNCH_CHECK();  // Macro for post kernel launch checks
                          // these checks are carried  ONLY IN DEBUG mode
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire