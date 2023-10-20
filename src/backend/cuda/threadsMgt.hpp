/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <common/dispatch.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace cuda {
// OVERALL USAGE (With looping):
// ...                                                      // OWN CODE
// threadsMgt<T> th(...);                                   // backend.hpp
// const dim3 threads{th.genThreads()};                     // backend.hpp
// const dim3 blocks{th.genBlocks(threads,..)};             // backend.hpp
// arrayfire::cuda::Kernel KER{GETKERNEL(..., th.loop0, th.loop1, th.loop2,
//                               th.loop3)};                // OWN CODE
// KER(threads,blocks,...);                                 // OWN CODE
// ...                                                      // OWN CODE
//
// OVERALL USAGE (without looping):
// ...                                                      // OWN CODE
// threadsMgt<T> th(...);                                   // backend.hpp
// const dim3 threads{th.genThreads()};                     // backend.hpp
// const dim3 blocks{th.genBlocksFull(threads,...)};        // backend.hpp
// arrayfire::cuda::Kernel KER{GETKERNEL(...)};             // OWN
// CODE KER(threads,blocks,...);                            // OWN CODE
// ...                                                      // OWN CODE
template<typename T>
class threadsMgt {
   public:
    bool loop0, loop1, loop2, loop3;

   private:
    const unsigned d0, d1, d2, d3;
    const T ndims;
    const unsigned maxParallelThreads;

   public:
    // INPUT: dims of the output array
    // INPUT: ndims of previous dims
    threadsMgt(const T dims[4], const T ndims);

    // Generate optimal thread values
    inline const dim3 genThreads() const;

    // INPUT threads, generated by genThreads()
    // OUTPUT blocks, supposing that each element results in 1 thread
    inline dim3 genBlocksFull(const dim3& threads) const;

    // Generate the optimal block values
    // INPUT threads, generated by genThreads()
    // INPUT nrInputs = number of input buffers read by kernel in parallel
    // INPUT nrOutputs = number of output buffers written by kernel in parallel
    // INPUT totalSize = size of all input arrays and all output arrays together
    // INPUT sizeofT = size of 1 element TO BE WRITTEN
    // OUTPUT blocks, assuming that the previously calculated loopings will be
    // executed in the kernel
    inline dim3 genBlocks(const dim3& threads, const unsigned nrInputs,
                          const unsigned nrOutputs, const size_t totalSize,
                          const size_t sizeofT);
};

// INPUT: dims of the output array
// INPUT: ndims of previous dims
template<typename T>
threadsMgt<T>::threadsMgt(const T dims[4], const T ndims)
    : loop0(false)
    , loop1(false)
    , loop2(false)
    , loop3(false)
    , d0(static_cast<unsigned>(dims[0]))
    , d1(static_cast<unsigned>(dims[1]))
    , d2(static_cast<unsigned>(dims[2]))
    , d3(static_cast<unsigned>(dims[3]))
    , ndims(ndims)
    , maxParallelThreads(getMaxParallelThreads(getActiveDeviceId())){};

// Generate optimal thread values
template<typename T>
const dim3 threadsMgt<T>::genThreads() const {
    // Performance is mainly dependend on:
    //    - reducing memory latency, by preferring a sequential read of
    //    cachelines (principally dim0)
    //    - more parallel threads --> higher occupation of available
    //    threads
    //    - more I/O operations per thread --> dims[3] indicates the #
    //    of I/Os handled by the kernel inside each thread, and outside
    //    the scope of the block scheduler
    // High performance is achievable with occupation rates as low as
    // 30%. Here we aim at 50%, to also cover older hardware with slower
    // cores.
    // https://stackoverflow.com/questions/7737772/improving-kernel-performance-by-increasing-occupancy
    // http://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf
    // https://www.cvg.ethz.ch/teaching/2011spring/gpgpu/GPU-Optimization.pdf
    // https://en.wikipedia.org/wiki/Graphics_Core_Next#SIMD_Vector_Unit

    // The performance for vectors is independent from array sizes.
    if ((d1 == 1) & (d2 == 1)) return dim3(128U);

    // TOTAL OCCUPATION = occup(dim0) * occup(dim1) * occup(dim2).
    // For linearized arrays, each linear block is allocated to a dim,
    // resulting in large numbers for dim0 & dim1.
    // - For dim2, we only return exact dividers of the array dim[3], so
    // occup(dim2)=100%
    // - For dim0 & dim1, we aim somewhere between 30% and 50%
    //      * Having 2 blocks filled + 1 thread in block 3 --> occup >
    //      2/3=66%
    //      * Having 3 blocks filled + 1 thread in block 4 --> occup >
    //      3/4=75%
    //      * Having 4 blocks filled + 1 thread in block 5 --> occup >
    //      4/5=80%
    constexpr unsigned OCCUPANCY_FACTOR{2U};  // at least 2 blocks filled

    // NVIDIA:
    //  warp             = 32
    //  possible blocks  = [32, 64, 96, 128, 160, 192, 224, 256, ..
    //  1024] best performance = [32, 64, 96, 128] optimal perf     =
    //  128; any combination
    //   NIVIDA always processes full wavefronts.  Allocating partial
    //   warps
    //   (<32) reduces throughput.  Performance reaches a plateau from
    //   128 with a slightly slowing for very large sizes.
    // For algorithm below:
    //  parallelThreads  = [32, 64, 96, 128]
    constexpr unsigned minThreads{32};
    const unsigned relevantElements{d0 * d1 * d2};
    constexpr unsigned warp{32};

    // For small array's, we reduce the maximum threads in 1 block to
    // improve parallelisme.  In worst case the scheduler can have 1
    // block per CU, even when only partly loaded. Range for block is:
    // [minThreads ... 4 * warp multiple]
    //   * NVIDIA: [4*32=128 threads]
    // At 4 * warp multiple, full wavefronts (queue of 4 partial
    // wavefronts) are all occupied.

    // We need at least maxParallelThreads to occupy all the CU's.
    const unsigned parallelThreads{
        relevantElements <= maxParallelThreads
            ? minThreads
            : std::min(4U, relevantElements / maxParallelThreads) * warp};

    // Priority 1: keep cachelines filled.  Aparrantly sharing
    // cachelines between CU's has a heavy cost. Testing confirmed that
    // the occupation is mostly > 50%
    const unsigned threads0{d0 == 1 ? 1
                            : d0 <= minThreads
                                ? minThreads  // better distribution
                                : std::min(128U, (divup(d0, warp) * warp))};

    // Priority 2: Fill the block, while respecting the occupation limit
    // (>66%) (through parallelThreads limit)
    const unsigned threads1{
        (threads0 * 64U <= parallelThreads) &&
                (!(d1 & (64U - 1U)) || (d1 > OCCUPANCY_FACTOR * 64U))
            ? 64U
        : (threads0 * 32U <= parallelThreads) &&
                (!(d1 & (32U - 1U)) || (d1 > OCCUPANCY_FACTOR * 32U))
            ? 32U
        : (threads0 * 16U <= parallelThreads) &&
                (!(d1 & (16U - 1U)) || (d1 > OCCUPANCY_FACTOR * 16U))
            ? 16U
        : (threads0 * 8U <= parallelThreads) &&
                (!(d1 & (8U - 1U)) || (d1 > OCCUPANCY_FACTOR * 8U))
            ? 8U
        : (threads0 * 4U <= parallelThreads) &&
                (!(d1 & (4U - 1U)) || (d1 > OCCUPANCY_FACTOR * 4U))
            ? 4U
        : (threads0 * 2U <= parallelThreads) &&
                (!(d1 & (2U - 1U)) || (d1 > OCCUPANCY_FACTOR * 2U))
            ? 2U
            : 1U};

    const unsigned threads01{threads0 * threads1};
    if ((d2 == 1) | (threads01 * 2 > parallelThreads))
        return dim3(threads0, threads1);

    // Priority 3: Only exact dividers are used, so that
    //  - overflow checking is not needed in the kernel.
    //  - occupation rate never is reduced
    // Chances are low that threads2 will be different from 1.
    const unsigned threads2{
        (threads01 * 8 <= parallelThreads) && !(d2 & (8U - 1U))   ? 8U
        : (threads01 * 4 <= parallelThreads) && !(d2 & (4U - 1U)) ? 4U
        : (threads01 * 2 <= parallelThreads) && !(d2 & (2U - 1U)) ? 2U
                                                                  : 1U};
    return dim3(threads0, threads1, threads2);
};

// INPUT threads, generated by genThreads()
// OUTPUT blocks, supposing that each element results in 1 thread
template<typename T>
inline dim3 threadsMgt<T>::genBlocksFull(const dim3& threads) const {
    const dim3 blocks{divup(d0, threads.x), divup(d1, threads.y),
                      divup(d2, threads.z)};
    return dim3(divup(d0, threads.x), divup(d1, threads.y),
                divup(d2, threads.z));
};

// Generate the optimal block values
// INPUT threads, generated by genThreads()
// INPUT nrInputs = number of input buffers read by kernel in parallel
// INPUT nrOutputs = number of output buffers written by kernel in parallel
// INPUT totalSize = size of all input arrays and all output arrays together
// INPUT sizeofT = size of 1 element TO BE WRITTEN
// OUTPUT blocks, assuming that the previously calculated loopings will be
// executed in the kernel
template<typename T>
inline dim3 threadsMgt<T>::genBlocks(const dim3& threads,
                                     const unsigned nrInputs,
                                     const unsigned nrOutputs,
                                     const size_t totalSize,
                                     const size_t sizeofT) {
    // The bottleneck of anykernel is dependent on the type of memory
    // used.
    // a) For very small arrays (elements < maxParallelThreads), each
    //  element receives it individual thread.
    // b) For arrays (in+out) smaller than 3/2 L2cache, memory access no
    //  longer is the bottleneck, because enough L2cache is available at any
    //  time. Threads are limited to reduce scheduling overhead.
    // c) For very large arrays and type sizes (<long double), 1 thread will
    //  not generate enough data to keep the memory sync mechanism
    //  saturated, so we start loooping inside each thread.
    dim3 blocks{1};
    const int activeDeviceId{getActiveDeviceId()};
    const unsigned* maxGridSize{
        reinterpret_cast<const unsigned*>(getMaxGridSize(activeDeviceId))};
    const size_t L2CacheSize{getL2CacheSize(activeDeviceId)};
    const unsigned cacheLine{getMemoryBusWidth(activeDeviceId)};
    const unsigned multiProcessorCount{getMultiProcessorCount(activeDeviceId)};
    const unsigned maxThreads{maxParallelThreads *
                              (sizeofT * nrInputs * nrInputs > 8 ? 1 : 2)};

    if (ndims == 1) {
        if (d0 > maxThreads) {
            if (totalSize * 2 > L2CacheSize * 3) {
                // General formula to calculate best #loops
                // Dedicated GPUs:
                //  32/sizeof(T)**2/#outBuffers*(3/4)**(#inBuffers-1)
                // Integrated GPUs:
                //  4/sizeof(T)/#outBuffers*(3/4)**(#inBuffers-1)
                unsigned largeVolDivider{cacheLine == 64
                                             ? sizeofT == 1   ? 4
                                               : sizeofT == 2 ? 2
                                                              : 1
                                             : (sizeofT == 1   ? 32
                                                : sizeofT == 2 ? 8
                                                               : 1) /
                                                   nrOutputs};
                for (unsigned i{1}; i < nrInputs; ++i)
                    largeVolDivider = largeVolDivider * 3 / 4;
                if (largeVolDivider > 1) {
                    blocks.x = d0 / (largeVolDivider * threads.x);
                    if (blocks.x == 0) blocks.x = 1;
                    loop0 = true;
                }
            } else {
                // A reduction to (1|2*)maxParallelThreads will be
                // performed
                blocks.x = maxThreads / threads.x;
                if (blocks.x == 0) blocks.x = 1;
                loop0 = true;
            }
        }
        if (!loop0) { blocks.x = divup(d0, threads.x); }
    } else {
        loop3    = d3 != 1;
        blocks.x = divup(d0, threads.x);
        blocks.z = divup(d2, threads.z);
        // contains the mandatory loops introduced by dim3 and dim2
        // gridSize overflow
        unsigned dim2and3Multiplier{d3};
        if (blocks.z > maxGridSize[2]) {
            dim2and3Multiplier = dim2and3Multiplier * blocks.z / maxGridSize[2];
            blocks.z           = maxGridSize[2];
            loop2              = true;
        }
        if ((d1 > threads.y) &
            (threads.x * blocks.x * d1 * threads.z * blocks.z > maxThreads)) {
            if ((d0 * sizeofT * 8 > cacheLine * multiProcessorCount) &
                (totalSize * 2 > L2CacheSize * 3)) {
                // General formula to calculate best #loops
                // Dedicated GPUs:
                //  32/sizeof(T)**2/#outBuffers*(3/4)**(#inBuffers-1)
                // Integrated GPUs:
                //  4/sizeof(T)/#outBuffers*(3/4)**(#inBuffers-1)
                unsigned largeVolDivider{
                    cacheLine == 64 ? sizeofT == 1   ? 4
                                      : sizeofT == 2 ? 2
                                                     : 1
                                    : (sizeofT == 1   ? 32
                                       : sizeofT == 2 ? 8
                                       : sizeofT == 4 ? 2
                                                      : 1) /
                                          (dim2and3Multiplier * nrOutputs)};
                for (unsigned i{1}; i < nrInputs; ++i)
                    largeVolDivider = largeVolDivider * 3 / 4;
                if (largeVolDivider > 1) {
                    blocks.y = d1 / (largeVolDivider * threads.y);
                    if (blocks.y == 0) blocks.y = 1;
                    loop1 = true;
                }
            } else {
                // A reduction to (1|2*)maxParallelThreads will be
                // performed
                blocks.y = maxThreads / (threads.x * blocks.x * threads.z *
                                         blocks.z * threads.y);
                if (blocks.y == 0) blocks.y = 1;
                loop1 = true;
            }
        }
        if (!loop1) { blocks.y = divup(d1, threads.y); }
        // Check on new overflows
        if (blocks.y > maxGridSize[1]) {
            blocks.y = maxGridSize[1];
            loop1    = true;
        }
    }

    return blocks;
};
}  // namespace cuda
}  // namespace arrayfire
