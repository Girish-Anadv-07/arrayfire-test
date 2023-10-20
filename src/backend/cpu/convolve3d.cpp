/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <arith.hpp>
#include <blas.hpp>
#include <common/defines.hpp>
#include <common/half.hpp>
#include <common/indexing_helpers.hpp>
#include <common/moddims.hpp>
#include <convolve.hpp>
#include <handle.hpp>
#include <kernel/convolve.hpp>
#include <platform.hpp>
#include <reorder.hpp>
#include <transpose.hpp>
#include <unwrap3d.hpp>
#include <wrap.hpp>
#include <vector>

#include <af/defines.h>
#include <af/dim4.hpp>

using af::dim4;
using arrayfire::common::flip;
using arrayfire::common::half;
using arrayfire::common::modDims;

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> convolve3_unwrap(const Array<T> &signal, const Array<T> &filter,
                          const dim4 &stride, const dim4 &padding,
                          const dim4 &dilation) {
    dim4 sDims = signal.dims();
    dim4 fDims = filter.dims();

    dim_t outputWidth =
        1 + (sDims[0] + 2 * padding[0] - (((fDims[0] - 1) * dilation[0]) + 1)) /
                stride[0];
    dim_t outputHeight =
        1 + (sDims[1] + 2 * padding[1] - (((fDims[1] - 1) * dilation[1]) + 1)) /
                stride[1];
    dim_t outputDepth =
        1 + (sDims[2] + 2 * padding[2] - (((fDims[2] - 1) * dilation[2]) + 1)) /
                stride[2];

    const bool retCols = false;
    Array<T> unwrapped =
        unwrap3d(signal, fDims[0], fDims[1], fDims[2], stride[0], stride[1], stride[2], padding[0],
               padding[1], padding[2], dilation[0], dilation[1], dilation[2], retCols);

    unwrapped  = reorder(unwrapped, dim4(1, 2, 0, 3));
    dim4 uDims = unwrapped.dims();
    unwrapped =
        modDims(unwrapped, dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));

    Array<T> collapsedFilter = flip(filter, {1, 1, 0, 0});
    collapsedFilter          = modDims(collapsedFilter,
                                       dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));

    Array<T> res =
        matmul(unwrapped, collapsedFilter, AF_MAT_TRANS, AF_MAT_NONE);
    res = modDims(res, dim4(outputWidth, outputHeight, signal.dims()[3],
                            outputDepth));
    Array<T> out = reorder(res, dim4(0, 1, 3, 2));

    return out;
}

template<typename T>
Array<T> convolve3(Array<T> const &signal, Array<T> const &filter,
                   const dim4 stride, const dim4 padding, const dim4 dilation) {
    Array<T> out = createEmptyArray<T>(dim4());
    out = convolve3_unwrap<T>(signal, filter, stride, padding, dilation);

    return out;
}

#define INSTANTIATE(T)                                                        \
    template Array<T> convolve3<T>(Array<T> const &signal,                    \
                                   Array<T> const &filter, const dim4 stride, \
                                   const dim4 padding, const dim4 dilation);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(half)
#undef INSTANTIATE

}  // namespace cpu
}  // namespace arrayfire
